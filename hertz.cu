/*
 * Contact decomposition of the hertz pairwise kernel. 
 * 
 * We take two datasets (from file) inputs (per-particle and contact data) and
 * expected output following a pairwise calculation.
 * 
 * We prepare an array of struct read-dataset and a struct of array
 * write-dataset. The kernel itself acts on individual contacts and writes out
 * *delta* [force] and [torque] values that must be postprocessed to give the
 * expected output.
 *
 * We sum through this indirection by first creating the inverse mapping and
 * running a separate gather kernel over the [force] and [torque] arrays.
 *
 */

//#define KERNEL_PRINT    //< debug printing in kernel
//#define MAP_BUILD_CHECK //< bounds and sanity checking in build_inverse_map
//#define AOS_EXTRA_DEBUG //< add (i,j) index information to struct

#ifdef KERNEL_PRINT
  #include "cuPrintf.cu"
#endif
#include "nvc_timer.h"
#include <fstream>
#include <iostream>
#include <string>
#include <assert.h>

#define ASSERT_NO_CUDA_ERROR( callReturningErrorstatus ) {     \
  cudaError_t err = callReturningErrorstatus;                  \
  if (err != cudaSuccess) {                                    \
    fprintf(stderr,                                            \
            "Cuda error (%s/%d) in file '%s' in line %i\n",    \
            cudaGetErrorString(err), err, __FILE__, __LINE__); \
    exit(1);                                                   \
  }                                                            \
} while(0);

using namespace std;

// --------------------------------------------------------------------------
// BUILDING AN INVERSE MAP
// --------------------------------------------------------------------------

int *exclusive_scan(int *array, int n) {
  int *result = new int[n];
  result[0] = 0;
  for (int i=1; i<n; i++) {
    result[i] = result[i-1] + array[i-1];
  }
  return result;
}

/*
 * Given a map[N] which implicitly defines a mapping from the set 
 * {0..(N-1)} to {0..(K-1)} 
 * we return a triplet representing the inverse mapping:
 *    - offset[K]
 *    - count[K]
 *    - imap[N]
 *
 * For all dest in {0..K}, offset[dest] and count[dest] give a range of 
 * indexes into imap that contains the src elements that point to 
 * dest in the given map.
 *
 */
void build_inverse_map(
  int *map, int N, int K,
  int *&offset, int *&count, int*&imap) {
#ifdef MAP_BUILD_CHECK
  assert(offset == NULL);
  assert(count == NULL);
  assert(imap == NULL);
#endif

  //output offset, count and imap
  count = new int[K];
  imap = new int[N];
  for (int i=0; i<K; i++) {
    count[i] = 0;
  }
  //use a temporary (sparse) inverse map so 
  //we only have to scan through map once
  int **sparse_imap = new int*[K];
  for (int i=0; i<K; i++) {
    sparse_imap[i] = new int[32];
  }

  //scan through the map and insert the inverse into sparse map
  for (int src=0; src<N; src++) {
    int dest = map[src];
#ifdef MAP_BUILD_CHECK
    assert(0 <= dest && dest < K);
    assert(count[dest] < 32);
#endif
    sparse_imap[dest][count[dest]] = src;
    count[dest]++;
  }

  //count used to calculate an offset into the packed imap
  offset = exclusive_scan(count, K);

  //squash sparse map into packed imap
  for (int dest=0; dest<K; dest++) {
    for (int i=0; i<count[dest]; i++) {
      assert((offset[dest]+i) < N);
      int src = sparse_imap[dest][i];
      imap[offset[dest]+i] = src;
    }
  }

#ifdef MAP_BUILD_CHECK
  //sanity check that the inverse map is correct
  for (int dest=0; dest<K; dest++) {
    for (int i=0; i<count[dest]; i++) {
      int src = imap[offset[dest]+i];
      if (map[src] != dest) {
        printf("map[%d] = %d != %d\n", src, map[src], dest);
      }
      assert(map[src] == dest);
    }
  }
#endif
}

// --------------------------------------------------------------------------
// UNPICKLING FROM FILE
// --------------------------------------------------------------------------

//datastructure from serialized data (input and expected_output)
struct params {
  //constants
  double dt;
  double nktv2p;
  int ntype;
  double *yeff;
  double *geff;
  double *betaeff;
  double *coeffFrict;

  //node data
  int nnode;
  double *x;
  double *v;
  double *omega;
  double *radius;
  double *mass;
  int    *type;
  double *force;
  double *torque;

  //edge data
  int nedge;
  int *edge;
  double *shear;
};

void print_params(struct params *p) {
  cout << "CONSTANTS" << endl;
  cout << "dt = " << p->dt << endl;
  cout << "nktv2p = " << p->nktv2p << endl;
  cout << "ntype = " << p->ntype << endl;
  for (int i=0; i<p->ntype*p->ntype; i++) {
    cout << "yeff[" << i << "] = " << p->yeff[i] << endl;
  }
  for (int i=0; i<p->ntype*p->ntype; i++) {
    cout << "geff[" << i << "] = " << p->geff[i] << endl;
  }
  for (int i=0; i<p->ntype*p->ntype; i++) {
    cout << "betaeff[" << i << "] = " << p->betaeff[i] << endl;
  }
  for (int i=0; i<p->ntype*p->ntype; i++) {
    cout << "coeffFrict[" << i << "] = " << p->coeffFrict[i] << endl;
  }

  cout << "NODES" << endl;
  cout << "nnode = " << p->nnode << endl;

  cout << "EDGES" << endl;
  cout << "nedge = " << p->nedge << endl;
}

//unpickle array
template<class T>
inline void fill_array(ifstream &file, T *array, int num_elements) {
  if (file.eof()) {
    cout << "Error unexpected eof!" << endl;
    exit(-1);
  }
  for (int i=0; i<num_elements; i++) {
    file >> array[i];
  }
}

//unpickle file
struct params *parse_file(string fname) {
  ifstream file (fname.c_str(), ifstream::in);
  if (!file.is_open()) {
    cout << "Could not open [" << fname << "]" << endl;
    exit(-1);
  }
  if (file.bad()) {
    cout << "Error with file [" << fname << "]" << endl;
    exit(-1);
  }

  struct params *result = new params;
  int ntype;
  int nnode;
  int nedge;

  file >> result->dt;
  file >> result->nktv2p;
  file >> ntype; result->ntype = ntype;
  result->yeff       = new double[ntype*ntype];
  result->geff       = new double[ntype*ntype];
  result->betaeff    = new double[ntype*ntype];
  result->coeffFrict = new double[ntype*ntype];
  fill_array(file, result->yeff,       (ntype*ntype));
  fill_array(file, result->geff,       (ntype*ntype));
  fill_array(file, result->betaeff,    (ntype*ntype));
  fill_array(file, result->coeffFrict, (ntype*ntype));

  file >> nnode; result->nnode = nnode;
  result->x      = new double[nnode*3];
  result->v      = new double[nnode*3];
  result->omega  = new double[nnode*3];
  result->radius = new double[nnode  ];
  result->mass   = new double[nnode  ];
  result->type   = new int[nnode];
  result->force  = new double[nnode*3];
  result->torque = new double[nnode*3];
  fill_array(file, result->x,      nnode*3);
  fill_array(file, result->v,      nnode*3);
  fill_array(file, result->omega,  nnode*3);
  fill_array(file, result->radius, nnode);
  fill_array(file, result->mass,   nnode);
  fill_array(file, result->type,   nnode);
  fill_array(file, result->force,  nnode*3);
  fill_array(file, result->torque, nnode*3);

  file >> nedge; result->nedge = nedge;
  result->edge = new int[nedge*2];
  result->shear = new double[nedge*3];
  fill_array(file, result->edge,  nedge*2);
  fill_array(file, result->shear, nedge*3);

  return result;
}

// --------------------------------------------------------------------------
// DEVICE KERNEL
// --------------------------------------------------------------------------

// AoS read-dataset for device
struct contact {
#ifdef AOS_EXTRA_DEBUG
  int i; int j;
#endif
  double xi[3];     double xj[3];
  double vi[3];     double vj[3];
  double omegai[3]; double omegaj[3];
  double radiusi;   double radiusj;
  double massi;     double massj;
  int    typei;     int typej;
};

#define sqrtFiveOverSix 0.91287092917527685576161630466800355658790782499663875
__global__ void aos_kernel(
  int ncontacts,
  struct contact *aos,
  double3 *force,
  double3 *torque, double3 *torquej,
  double  *shear) {
  //TODO(0): don't hardcode, push these into constant memory
  double dt = 0.00001;
  double nktv2p = 1;
  double yeff = 3134796.2382445144467056;
  double geff = 556173.5261401557363570;
  double betaeff = -0.3578571305033167;
  double coeffFrict = 0.5;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < ncontacts) {
    struct contact c = aos[idx];
#if 0
    cuPrintf("idx = %d\n", idx);
    cuPrintf("xi = {%f, %f, %f}\n",
      c.xi[0], c.xi[1], c.xi[2]);
    cuPrintf("xj = {%f, %f, %f}\n",
      c.xj[0], c.xj[1], c.xj[2]);
    cuPrintf("vi = {%f, %f, %f}\n",
      c.vi[0], c.vi[1], c.vi[2]);
    cuPrintf("vj = {%f, %f, %f}\n",
      c.vj[0], c.vj[1], c.vj[2]);
    cuPrintf("omegai = {%f, %f, %f}\n",
      c.omegai[0], c.omegai[1], c.omegai[2]);
    cuPrintf("omegaj = {%f, %f, %f}\n",
      c.omegaj[0], c.omegai[1], c.omegai[2]);
    cuPrintf("radiusi = %f\n", c.radiusi);
    cuPrintf("radiusj = %f\n", c.radiusj);
    cuPrintf("massi = %f\n", c.massi);
    cuPrintf("massj = %f\n", c.massj);
    cuPrintf("typei = %d\n", c.typei);
    cuPrintf("typej = %d\n", c.typej);
    cuPrintf("force = {%f, %f, %f}\n",
      force[idx].x, force[idx].y, force[idx].z);
    cuPrintf("torque = {%f, %f, %f}\n",
      torque[idx].x, torque[idx].y, torque[idx].z);
    cuPrintf("torquej = {%f, %f, %f}\n",
      torquej[idx].x, torquej[idx].y, torquej[idx].z);
    cuPrintf("shear = {%f, %f, %f}\n",
      shear[(idx*3)], shear[(idx*3)+1], shear[(idx*3)+2]);
#endif

    // del is the vector from j to i
    double delx = c.xi[0] - c.xj[0];
    double dely = c.xi[1] - c.xj[1];
    double delz = c.xi[2] - c.xj[2];

    double rsq = delx*delx + dely*dely + delz*delz;
    double radsum = c.radiusi + c.radiusj;
    if (rsq >= radsum*radsum) {
      //unset non-touching atoms
      shear[(idx*3)  ] = 0.0;
      shear[(idx*3)+1] = 0.0;
      shear[(idx*3)+2] = 0.0;
    } else {
      //distance between centres of atoms i and j
      //or, magnitude of del vector
      double r = sqrt(rsq);
      double rinv = 1.0/r;
      double rsqinv = 1.0/rsq;

      // relative translational velocity
      double vr1 = c.vi[0] - c.vj[0];
      double vr2 = c.vi[1] - c.vj[1];
      double vr3 = c.vi[2] - c.vj[2];

      // normal component
      double vnnr = vr1*delx + vr2*dely + vr3*delz;
      double vn1 = delx*vnnr * rsqinv;
      double vn2 = dely*vnnr * rsqinv;
      double vn3 = delz*vnnr * rsqinv;

      // tangential component
      double vt1 = vr1 - vn1;
      double vt2 = vr2 - vn2;
      double vt3 = vr3 - vn3;

      // relative rotational velocity
      double wr1 = (c.radiusi*c.omegai[0] + c.radiusj*c.omegaj[0]) * rinv;
      double wr2 = (c.radiusi*c.omegai[1] + c.radiusj*c.omegaj[1]) * rinv;
      double wr3 = (c.radiusi*c.omegai[2] + c.radiusj*c.omegaj[2]) * rinv;

      // normal forces = Hookian contact + normal velocity damping
      double meff = c.massi*c.massj/(c.massi+c.massj);
      //not-implemented: freeze_group_bit

      double deltan = radsum-r;

      //derive contact model parameters (inlined)
      //yeff, geff, betaeff, coeffFrict are constant lookup tables
      double reff = c.radiusi * c.radiusj / (c.radiusi + c.radiusj);
      double sqrtval = sqrt(reff * deltan);
      double Sn = 2.    * yeff * sqrtval;
      double St = 8.    * geff * sqrtval;
      double kn = 4./3. * yeff * sqrtval;
      double kt = St;
      double gamman=-2.*sqrtFiveOverSix*betaeff*sqrt(Sn*meff);
      double gammat=-2.*sqrtFiveOverSix*betaeff*sqrt(St*meff);
      double xmu=coeffFrict;
      //not-implemented if (dampflag == 0) gammat = 0;
      kn /= nktv2p;
      kt /= nktv2p;

      double damp = gamman*vnnr*rsqinv;
      double ccel = kn*(radsum-r)*rinv - damp;

      //not-implemented cohesionflag

      // relative velocities
      double vtr1 = vt1 - (delz*wr2-dely*wr3);
      double vtr2 = vt2 - (delx*wr3-delz*wr1);
      double vtr3 = vt3 - (dely*wr1-delx*wr2);

      // shear history effects
      shear[(idx*3)  ] += vtr1 * dt;
      shear[(idx*3)+1] += vtr2 * dt;
      shear[(idx*3)+2] += vtr3 * dt;

      // rotate shear displacements
      double rsht = shear[(idx*3)  ]*delx + 
                    shear[(idx*3)+1]*dely + 
                    shear[(idx*3)+2]*delz;
      rsht *= rsqinv;

      shear[(idx*3)  ] -= rsht*delx;
      shear[(idx*3)+1] -= rsht*dely;
      shear[(idx*3)+2] -= rsht*delz;

      // tangential forces = shear + tangential velocity damping
      double fs1 = - (kt*shear[(idx*3)  ] + gammat*vtr1);
      double fs2 = - (kt*shear[(idx*3)+1] + gammat*vtr2);
      double fs3 = - (kt*shear[(idx*3)+2] + gammat*vtr3);

      // rescale frictional displacements and forces if needed
      double fs = sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
      double fn = xmu * fabs(ccel*r);
      double shrmag = 0;
      if (fs > fn) {
        shrmag = sqrt(shear[(idx*3)  ]*shear[(idx*3)  ] +
                      shear[(idx*3)+1]*shear[(idx*3)+1] +
                      shear[(idx*3)+2]*shear[(idx*3)+2]);
        if (shrmag != 0.0) {
          shear[(idx*3)  ] = (fn/fs) * (shear[(idx*3)  ] + gammat*vtr1/kt) - gammat*vtr1/kt;
          shear[(idx*3)+1] = (fn/fs) * (shear[(idx*3)+1] + gammat*vtr2/kt) - gammat*vtr2/kt;
          shear[(idx*3)+2] = (fn/fs) * (shear[(idx*3)+2] + gammat*vtr3/kt) - gammat*vtr3/kt;
          fs1 *= fn/fs;
          fs2 *= fn/fs;
          fs3 *= fn/fs;
        } else {
          fs1 = fs2 = fs3 = 0.0;
        }
      }

      double fx = delx*ccel + fs1;
      double fy = dely*ccel + fs2;
      double fz = delz*ccel + fs3;

      double tor1 = rinv * (dely*fs3 - delz*fs2);
      double tor2 = rinv * (delz*fs1 - delx*fs3);
      double tor3 = rinv * (delx*fs2 - dely*fs1);

      // this is what we've been working up to!
      force[idx].x = fx;
      force[idx].y = fy;
      force[idx].z = fz;

      torque[idx].x -= c.radiusi*tor1;
      torque[idx].y -= c.radiusi*tor2;
      torque[idx].z -= c.radiusi*tor3;

      torquej[idx].x -= c.radiusj*tor1;
      torquej[idx].y -= c.radiusj*tor2;
      torquej[idx].z -= c.radiusj*tor3;

#if 0
      cuPrintf("force' = {%f, %f, %f}\n",
        force[idx].x, force[idx].y, force[idx].z);
      cuPrintf("torque' = {%f, %f, %f}\n",
        torque[idx].x, torque[idx].y, torque[idx].z);
      cuPrintf("torquej' = {%f, %f, %f}\n",
        torquej[idx].x, torquej[idx].y, torquej[idx].z);
      cuPrintf("shear' = {%f, %f, %f}\n",
        shear[(idx*3)], shear[(idx*3)+1], shear[(idx*3)+2]);
#endif
    }
  }
}

__global__ void gather_kernel(
  int nparticles,
  double3 *force_delta, double3 *torquei_delta, double3 *torquej_delta,
  int *ioffset, int *icount, int *imapinv,
  int *joffset, int *jcount, int *jmapinv,
  //outputs
  double *force, double *torque) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nparticles) {
    double fdelta[3] = {0.0, 0.0, 0.0};
    double tdelta[3] = {0.0, 0.0, 0.0};

    int ioff = ioffset[idx];
    for (int i=0; i<icount[idx]; i++) {
      int e = imapinv[ioff+i];
      fdelta[0] += force_delta[e].x;
      fdelta[1] += force_delta[e].y;
      fdelta[2] += force_delta[e].z;

      tdelta[0] += torquei_delta[e].x;
      tdelta[1] += torquei_delta[e].y;
      tdelta[2] += torquei_delta[e].z;
    }

    int joff = joffset[idx];
    for (int i=0; i<jcount[idx]; i++) {
      int e = jmapinv[joff+i];

      fdelta[0] -= force_delta[e].x;
      fdelta[1] -= force_delta[e].y;
      fdelta[2] -= force_delta[e].z;

      tdelta[0] += torquej_delta[e].x;
      tdelta[1] += torquej_delta[e].y;
      tdelta[2] += torquej_delta[e].z;
    }

    //output
    force[(idx*3)]   += fdelta[0];
    force[(idx*3)+1] += fdelta[1];
    force[(idx*3)+2] += fdelta[1];

    torque[(idx*3)]   += tdelta[0];
    torque[(idx*3)+1] += tdelta[1];
    torque[(idx*3)+2] += tdelta[1];
  }
}

// --------------------------------------------------------------------------
// HOST SIDE PRE AND POST PROCESS
// --------------------------------------------------------------------------

//check two vectors against each other (using absolute difference of elements)
bool check_result_vector(const char* id, double expected[3], double actual[3], const double epsilon) {
  static bool verbose = false;
  bool flag = (fabs(expected[0] - actual[0]) > epsilon ||
               fabs(expected[1] - actual[1]) > epsilon ||
               fabs(expected[2] - actual[2]) > epsilon);
  const char *marker = flag ? "***" : "   ";

  if (flag && verbose) {
    printf("%s%s: {%.16f, %.16f, %.16f} / {%.16f, %.16f, %.16f}%s\n",
        marker,
        id,
        expected[0], expected[1], expected[2],
        actual[0], actual[1], actual[2],
        marker
        );
  }
  return flag;
}

static NVCTimer *timers = new NVCTimer[8];

/*
 * Run [num_iter] iterations of the hertz computation and return the total time
 * (in milliseconds) for the per-iteration cost. NB: the returned time does not
 * include one-time costs.
 */
double array_of_struct(int argc, char **argv,
    struct params *input, struct params *expected_output,
    int num_iter) {

  for (int i=0; i<8; i++) {
    timers[i].init();
  }
  timers[0].set_name("AoS memcpy to dev");
  timers[1].set_name("Pairwise kernel");
  timers[2].set_name("Gather kernel");
  timers[3].set_name("Result fetch");

  timers[4].set_name("AoS generate");
  timers[5].set_name("Malloc cuda datastructures");
  timers[6].set_name("Build inverse mappings for gather kernel");

  //--------------------
  // One-time only costs
  //--------------------

  //AoS generate
  timers[4].start();
  struct contact *aos = new contact[input->nedge];
  for (int e=0; e<input->nedge; e++) {
    int i = input->edge[(e*2)];
    int j = input->edge[(e*2)+1];

#ifdef AOS_EXTRA_DEBUG
    aos[e].i = i;
    aos[e].j = j;
#endif

    aos[e].xi[0] = input->x[(i*3)];
    aos[e].xi[1] = input->x[(i*3)+1];
    aos[e].xi[2] = input->x[(i*3)+2];
    aos[e].xj[0] = input->x[(j*3)];
    aos[e].xj[1] = input->x[(j*3)+1];
    aos[e].xj[2] = input->x[(j*3)+2];

    aos[e].vi[0] = input->v[(i*3)];
    aos[e].vi[1] = input->v[(i*3)+1];
    aos[e].vi[2] = input->v[(i*3)+2];
    aos[e].vj[0] = input->v[(j*3)];
    aos[e].vj[1] = input->v[(j*3)+1];
    aos[e].vj[2] = input->v[(j*3)+2];

    aos[e].omegai[0] = input->omega[(i*3)];
    aos[e].omegai[1] = input->omega[(i*3)+1];
    aos[e].omegai[2] = input->omega[(i*3)+2];
    aos[e].omegaj[0] = input->omega[(j*3)];
    aos[e].omegaj[1] = input->omega[(j*3)+1];
    aos[e].omegaj[2] = input->omega[(j*3)+2];

    aos[e].radiusi = input->radius[i];
    aos[e].radiusj = input->radius[j];

    aos[e].massi = input->mass[i];
    aos[e].massj = input->mass[j];

    aos[e].typei = input->type[i];
    aos[e].typej = input->type[j];
  }

  struct contact *d_aos;
  const int d_aos_size = input->nedge * sizeof(struct contact);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_aos, d_aos_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_aos, aos, d_aos_size, cudaMemcpyHostToDevice));
  timers[4].stop();
  timers[4].add_to_total();

  timers[5].start();
  double3 *d_force_delta;
  double3 *d_torquei_delta;
  double3 *d_torquej_delta;
  const int d_delta_size = input->nedge * sizeof(double3);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_force_delta, d_delta_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_torquei_delta, d_delta_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_torquej_delta, d_delta_size));

  double *d_force;
  double *d_torque;
  const int d_output_size = input->nnode * 3 * sizeof(double);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_force, d_output_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_torque, d_output_size));

  double *d_shear;
  const int d_shear_size = input->nedge * 3 * sizeof(double);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_shear, d_shear_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_shear, input->shear, d_shear_size, cudaMemcpyHostToDevice));
  timers[5].stop();
  timers[5].add_to_total();

  timers[6].start();
  //inverse mappings for (i,j) particle pairs
  int *imap = new int[input->nedge];
  int *jmap = new int[input->nedge];
  for (int e=0; e<input->nedge; e++) {
    imap[e] = input->edge[(e*2)  ];
    jmap[e] = input->edge[(e*2)+1];
  }
  int *ioffset = NULL;
  int *icount = NULL;
  int *imapinv = NULL;
  int *joffset = NULL;
  int *jcount = NULL;
  int *jmapinv = NULL;
  build_inverse_map(imap, input->nedge, input->nnode,
    ioffset, icount, imapinv);
  build_inverse_map(jmap, input->nedge, input->nnode,
    joffset, jcount, jmapinv);

  int *d_ioffset;
  int *d_icount;
  int *d_imapinv;
  int *d_joffset;
  int *d_jcount;
  int *d_jmapinv;
  const int d_nnode_size = input->nnode * sizeof(int);
  const int d_nedge_size = input->nedge * sizeof(int);
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_ioffset, d_nnode_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_icount, d_nnode_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_imapinv, d_nedge_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_joffset, d_nnode_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_jcount, d_nnode_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_jmapinv, d_nedge_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_ioffset, ioffset, d_nnode_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_icount, icount, d_nnode_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_imapinv, imapinv, d_nedge_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_joffset, joffset, d_nnode_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_jcount, jcount, d_nnode_size, cudaMemcpyHostToDevice));
  ASSERT_NO_CUDA_ERROR(
    cudaMemcpy(d_jmapinv, jmapinv, d_nedge_size, cudaMemcpyHostToDevice));
  timers[6].stop();
  timers[6].add_to_total();

  //TODO(1): copy real x, v, omega in PREPROCESS
  //These are dummy structures just for timing
  double *d_fake_x;
  double *d_fake_v;
  double *d_fake_omega;
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fake_x, d_output_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fake_v, d_output_size));
  ASSERT_NO_CUDA_ERROR(
    cudaMalloc((void **)&d_fake_omega, d_output_size));

  //--------------------
  // Per-iteration costs
  //--------------------
  double time = 0.0; //total time in milliseconds for num_iter iterations
  for (int run=0; run<num_iter; run++) {
    //PREPROCESSING
    //copy across structures that change between kernel invocations, 
    //reset delta structures (force/torque).
    timers[0].start();
    //TODO(1): just copy dummy structures for timing
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_x, input->x, d_output_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_v, input->v, d_output_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_fake_omega, input->omega, d_output_size, cudaMemcpyHostToDevice));

    ASSERT_NO_CUDA_ERROR(
      cudaMemset((void *)d_force_delta, 0, d_delta_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMemset((void *)d_torquei_delta, 0, d_delta_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMemset((void *)d_torquej_delta, 0, d_delta_size));
    timers[0].stop();

    //NB: safe to omit from preprocess costs because they do not change between
    //kernel invocations.
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_force, input->force, d_output_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_torque, input->torque, d_output_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_shear, input->shear, d_shear_size, cudaMemcpyHostToDevice));

    //-----------------------------------------------------------------------

    //KERNEL INVOCATION
    //pairwise kernel computes delta values (force/torque) and shear
    //gather kernel produces force/torque results
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Pre-kernel error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

#ifdef KERNEL_PRINT
    cudaPrintfInit();
#endif

    const int blockSize = 128;
    dim3 gridSize((input->nedge / blockSize)+1);
    timers[1].start();
    aos_kernel<<<gridSize, blockSize>>>(
      input->nedge,
      d_aos,
      d_force_delta, d_torquei_delta, d_torquej_delta, d_shear);
    timers[1].stop();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Post-kernel error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

    const int gatherBlockSize = 128;
    dim3 gatherGridSize((input->nnode / gatherBlockSize)+1);
    timers[2].start();
    gather_kernel<<<gatherGridSize, gatherBlockSize>>>(
      input->nnode,
      d_force_delta, d_torquei_delta, d_torquej_delta,
      d_ioffset, d_icount, d_imapinv,
      d_joffset, d_jcount, d_jmapinv,
      d_force, d_torque);
    timers[2].stop();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Post-gather error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

    cudaThreadSynchronize();
#ifdef KERNEL_PRINT
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif

    //-----------------------------------------------------------------------

    //POSTPROCESSING
    //memcpy data back to host
    timers[3].start();
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(input->force, d_force, d_output_size, cudaMemcpyDeviceToHost));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(input->torque, d_torque, d_output_size, cudaMemcpyDeviceToHost));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(input->shear, d_shear, d_shear_size, cudaMemcpyDeviceToHost));
    timers[3].stop();

    //-----------------------------------------------------------------------

    //CHECKING
    //only check results the first time around
    if (run == 0) {
      const double epsilon = 0.000001;
      int nerror = 0;
      for (int n=0; n<input->nnode; n++) {
        bool flag =
          check_result_vector(
            "force",
            &expected_output->force[(n*3)], &input->force[(n*3)], epsilon) ||
          check_result_vector(
            "torque",
            &expected_output->torque[(n*3)], &input->torque[(n*3)], epsilon);

        if (flag) {
          nerror++;
        }
      }
      printf("Warning: force/torque errors: %d\n", nerror);

      for (int n=0; n<input->nedge; n++) {
        bool flag = check_result_vector(
          "shear", &expected_output->shear[(n*3)], &input->shear[(n*3)], 0.000001);

        if (flag) {
          exit(-1);
        }
      }
    }

    timers[0].add_to_total();
    timers[1].add_to_total();
    timers[2].add_to_total();
    timers[3].add_to_total();
  }

  //CLEANUP
  cudaFree(d_aos);
  cudaFree(d_force_delta);
  cudaFree(d_torquei_delta);
  cudaFree(d_torquej_delta);
  cudaFree(d_shear);
  cudaFree(d_force);
  cudaFree(d_torque);

  cudaFree(d_ioffset);
  cudaFree(d_icount);
  cudaFree(d_imapinv);
  cudaFree(d_joffset);
  cudaFree(d_jcount);
  cudaFree(d_jmapinv);

  //TODO(1): free dummy structures
  cudaFree(d_fake_x);
  cudaFree(d_fake_v);
  cudaFree(d_fake_omega);

  //TIMING
  printf("One time costs\n");
  printf("--------------\n");
  for (int i=4; i<7; i++) {
    printf("%d [%s] %.1fms\n", i, timers[i].get_name().c_str(), timers[i].total_time());
  }
  printf("Timer breakdown\n");
  printf("---------------\n");
  for (int i=0; i<4; i++) {
    printf("%d [%s] %.1fms\n", i, timers[i].get_name().c_str(), timers[i].total_time());
    time += timers[i].total_time();
  }

  return time;
}

// --------------------------------------------------------------------------
// MAIN PROGRAM
// --------------------------------------------------------------------------

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <input_file> <expected_output_file> [num_iterations]\n", argv[0]);
    exit(-1);
  }

  string input_filename(argv[1]);
  struct params *input = parse_file(input_filename);

  string expected_output_filename(argv[2]);
  struct params *expected_output = parse_file(expected_output_filename);

  int num_iter = 1000;
  if (argc > 3) {
    num_iter = atoi(argv[3]);
  }

  printf("Input: %s\n", argv[1]);
  printf("Expected Output: %s\n", argv[2]);
  printf("Num Iterations: %d\n", num_iter);

  double time = array_of_struct(argc, argv, input, expected_output, num_iter);
  printf("Total time(ms): %f\n", time);
  printf("Time per iteration(ms): %f\n", (time/num_iter));

  return 0;
}
