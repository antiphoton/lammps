/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:  Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include <complex>
#include <string.h>
#include <stdlib.h>
#include "compute_diamondorder_atom.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"


using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

ComputeDiamondOrderAtom::ComputeDiamondOrderAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 3 ) error->all(FLERR,"Illegal compute diamondorder/atom command");

  ndegree = 6;
  nnn = 6;
  cutsq = 0.0;
  rsoft=0;
  devSq=-1;
  oxygenId=-1;
  hydrogenId=-1;


  // process optional args

  int iarg = 3;
  while (iarg < narg) {
      if (strcmp(arg[iarg],"degree") == 0) {
          if (iarg+2 > narg) error->all(FLERR,"Illegal compute diamondorder/atom command");
          ndegree = force->numeric(FLERR,arg[iarg+1]);
          if (ndegree < 0)
              error->all(FLERR,"Illegal compute diamondorder/atom command");
          iarg += 2;
      } else if (strcmp(arg[iarg],"rsoft") == 0) {
          if (iarg+2 > narg) error->all(FLERR,"Illegal compute diamondorder/atom command");
          rsoft = force->numeric(FLERR,arg[iarg+1]);
          if (rsoft <= 0)
              error->all(FLERR,"Illegal compute diamondorder/atom command");
          iarg += 2;
      } else if (strcmp(arg[iarg],"nnn") == 0) {
          if (iarg+2 > narg) error->all(FLERR,"Illegal compute diamondorder/atom command");
          if (strcmp(arg[iarg+1],"NULL") == 0) 
              nnn = 0;
          else {
              nnn = force->numeric(FLERR,arg[iarg+1]);
              if (nnn < 0)
                  error->all(FLERR,"Illegal compute diamondorder/atom command");
          }
          iarg += 2;
      } else if (strcmp(arg[iarg],"cutoff") == 0) {
          if (iarg+2 > narg) error->all(FLERR,"Illegal compute diamondorder/atom command");
          double cutoff = force->numeric(FLERR,arg[iarg+1]);
          if (cutoff <= 0.0)
              error->all(FLERR,"Illegal compute diamondorder/atom command");
          cutsq = cutoff*cutoff;
          iarg += 2;
      } else if (strcmp(arg[iarg],"hydrogen") == 0) {
          if (iarg+2 > narg) error->all(FLERR,"Illegal compute diamondorder/atom command");
          hydrogenId = force->numeric(FLERR,arg[iarg+1]);
          iarg += 2;
      } else if (strcmp(arg[iarg],"oxygen") == 0) {
          if (iarg+2 > narg) error->all(FLERR,"Illegal compute diamondorder/atom command");
          oxygenId = force->numeric(FLERR,arg[iarg+1]);
          iarg += 2;
      } else if (strcmp(arg[iarg],"deviation") == 0) {
          if (iarg+2 > narg) error->all(FLERR,"Illegal compute diamondorder/atom command");
          double deviation = force->numeric(FLERR,arg[iarg+1]);
          if (deviation < 0.0)
              error->all(FLERR,"Illegal compute diamondorder/atom command");
          devSq = deviation*deviation;
          iarg += 2;
      } else error->all(FLERR,"Illegal compute diamondorder/atom command");
  }

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  comm_forward=1;
  qlmarray=NULL;
  qnvector = NULL;
  maxneigh = 0;
  distsq = NULL;
  nearest = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeDiamondOrderAtom::~ComputeDiamondOrderAtom()
{
  memory->destroy(qlmarray);
  memory->destroy(qnvector);
  memory->destroy(distsq);
  memory->destroy(nearest);
}

/* ---------------------------------------------------------------------- */

void ComputeDiamondOrderAtom::init()
{
  if (force->pair == NULL)
    error->all(FLERR,"Compute diamondorder/atom requires a pair style be defined");
  if (cutsq == 0.0) cutsq = force->pair->cutforce * force->pair->cutforce;
  else if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR, "Compute diamondorder/atom cutoff is longer than pairwise cutoff");
  if (rsoft<0) {
      error->all(FLERR, "Compute diamondorder/atom rsoft is negative");
  }

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"diamondorder/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute diamondorder/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeDiamondOrderAtom::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */


int ComputeDiamondOrderAtom::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
    int i,j,m;
    m=0;
    for (i=0;i<n;i++) {
        j=list[i];
        buf[m++]=qlmarray[j][i_comm];
    }
    return m;
}
void ComputeDiamondOrderAtom::unpack_forward_comm(int n, int first, double *buf) {
    int i,m,last;
    m=0;
    last=first+n;
    for (i=first;i<last;i++) {
        qlmarray[i][i_comm]=buf[m++];
    }
}

void ComputeDiamondOrderAtom::compute_peratom()
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *iH=new int[4];

  invoked_peratom = update->ntimestep;

  // grow order parameter array if necessary

  if (atom->nlocal > nmax) {
    memory->destroy(qlmarray);
    memory->destroy(qnvector);
    nmax = atom->nmax;
    memory->create(qlmarray,nmax,(2*ndegree+1)*2,"diamondorder/atom:qlmarray");
    memory->create(qnvector,nmax,"diamondorder/atom:qnvector");
    vector_atom = qnvector;
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute order parameter for each atom in group
  // use full neighbor list to count atoms less than cutoff

  double **x = atom->x;
  int *mask = atom->mask;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (oxygenId>=0&&atom->type[i]!=oxygenId) {
        continue;
    }
    double *qlm= qlmarray[i];
    if (mask[i] & groupbit) {
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      jlist = firstneigh[i];
      jnum = numneigh[i];
      
      // insure distsq and nearest arrays are long enough

      if (jnum > maxneigh) {
        memory->destroy(distsq);
        memory->destroy(nearest);
        maxneigh = jnum;
        memory->create(distsq,maxneigh,"diamondorder/atom:distsq");
        memory->create(nearest,maxneigh,"diamondorder/atom:nearest");
      }

      // loop over list of all neighbors within force cutoff
      // distsq[] = distance sq to each
      // nearest[] = atom indices of neighbors

      iH[0]=atom->map(atom->tag[i]+1);
      iH[1]=atom->map(atom->tag[i]+2);

      int ncount = 0;
      double sWeight=0;
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        if (oxygenId>=0&&atom->type[j]!=oxygenId) {
            continue;
        }

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];

        iH[2]=atom->map(atom->tag[j]+1);
        iH[3]=atom->map(atom->tag[j]+2);

        if (!hydrogenBond(xtmp,ytmp,ztmp,delx,dely,delz,iH)) {
            continue;
        }

        rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < cutsq) {
          distsq[ncount] = rsq;
          nearest[ncount++] = j;
        }
      }

      // if not nnn neighbors, order parameter = 0;

      if (ncount==0) {
          for (int m=0;m<2*ndegree+1;m++) {
              qlm[m*2]=0;
              qlm[m*2+1]=0;
          }
          continue;
      }
      if (ncount < nnn) {
	//qn[0] = qn[1] = 0.0;
        //continue;
      }

      // if nnn > 0, use only nearest nnn neighbors

      if (nnn > 0 && nnn<ncount) {

          select2(nnn,ncount,distsq,nearest);
          ncount = nnn;
      }
      for (int m=0;m<2*ndegree+1;m++) {
          qlm[m*2]=0;
          qlm[m*2+1]=0;
          double sWeight=0;

          for (jj = 0; jj < ncount; jj++) {
            j = nearest[jj];
            j &= NEIGHMASK;
            
            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            double r=sqrt(delx*delx+dely*dely+delz*delz);
            double rinv = 1.0/r;
            double weight=smearing(r);
            delx*=rinv;
            dely*=rinv;
            delz*=rinv;
            add_qlm_complex(m-ndegree,weight,delx,dely,delz,qlm+m*2,qlm+m*2+1);
            sWeight+=weight;
          }
          if (ncount>0) {
              qlm[m*2]/=sWeight;
              qlm[m*2+1]/=sWeight;
          }
      }
    }
  }
  for (i_comm=0;i_comm<2*(ndegree*2+1);i_comm++) {
      comm->forward_comm_compute(this);
  }
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (oxygenId>=0&&atom->type[i]!=oxygenId) {
        continue;
    }
    if (mask[i] & groupbit) {
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      jlist = firstneigh[i];
      jnum = numneigh[i];
      
      // insure distsq and nearest arrays are long enough

      if (jnum > maxneigh) {
        memory->destroy(distsq);
        memory->destroy(nearest);
        maxneigh = jnum;
        memory->create(distsq,maxneigh,"diamondorder/atom:distsq");
        memory->create(nearest,maxneigh,"diamondorder/atom:nearest");
      }

      // loop over list of all neighbors within force cutoff
      // distsq[] = distance sq to each
      // nearest[] = atom indices of neighbors

      int ncount = 0;
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        if (oxygenId>=0&&atom->type[j]!=oxygenId) {
            continue;
        }

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < cutsq) {
          distsq[ncount] = rsq;
          nearest[ncount++] = j;
        }
      }

      // if not nnn neighbors, order parameter = 0;

      if (ncount < nnn) {
	//qn[0] = qn[1] = 0.0;
        //continue;
      }

      // if nnn > 0, use only nearest nnn neighbors

      if (nnn > 0&&nnn<ncount) {
          select2(nnn,ncount,distsq,nearest);
          ncount = nnn;
      }

      double usum = 0.0;
      double vsum = 0.0;
      double sWeight=0;

      for (jj = 0; jj < ncount; jj++) {
          j = nearest[jj];
          j &= NEIGHMASK;
          double delx=atom->x[j][0]-atom->x[i][0];
          double dely=atom->x[j][1]-atom->x[i][1];
          double delz=atom->x[j][2]-atom->x[i][2];
          double weight=smearing(sqrt(delx*delx+dely*dely+delz*delz));
          add_qn_complex(i, j, weight, &usum, &vsum);
          sWeight+=weight;
      }
      if (ncount>0) {
          usum/=sWeight;
          vsum/=sWeight;
      }
      qnvector[i]=usum;
    }
  }
  delete[] iH;
}

void ComputeDiamondOrderAtom::add_qlm_complex(int m,double frr,double x,double y,double z,double *u,double *v) {
    double f=frr*polar_prefactor(ndegree,m,z);
    double phi=atan2(y,x);
    phi*=m;
    *u += f*cos(phi);
    *v += f*sin(phi);
}

// calculate order parameter using std::complex::pow function

inline void ComputeDiamondOrderAtom::add_qn_complex(int i,int j,double frr, double *u, double *v) {
    double x=0,y=0;
    double normI=0,normJ=0;
    int m;
    for (m=0;m<2*ndegree+1;m++) {
        double xI=qlmarray[i][m*2],yI=qlmarray[i][m*2+1];
        double xJ=qlmarray[j][m*2],yJ=-qlmarray[j][m*2+1];
        double xD=xI*xJ-yI*yJ;
        double yD=xI*yJ+xJ*yI;
        double nID=xI*xI+yI*yI;
        double nJD=xJ*xJ+yJ*yJ;
        x+=xD;
        y+=yD;
        normI+=nID;
        normJ+=nJD;
    }
    double normIJ=sqrt(normI*normJ);
    if (normIJ>0) {
        *u+=x/normIJ*frr;
        *v+=y/normIJ*frr;
    }
}

/* ----------------------------------------------------------------------
   select2 routine from Numerical Recipes (slightly modified)
   find k smallest values in array of length n
   sort auxiliary array at same time
------------------------------------------------------------------------- */

#define SWAP(a,b)   tmp = a; a = b; b = tmp;
#define ISWAP(a,b) itmp = a; a = b; b = itmp;

/* ---------------------------------------------------------------------- */

void ComputeDiamondOrderAtom::select2(int k, int n, double *arr, int *iarr)
{
  int i,ir,j,l,mid,ia,itmp;
  double a,tmp;

  arr--;
  iarr--;
  l = 1;
  ir = n;
  for (;;) {
    if (ir <= l+1) {
      if (ir == l+1 && arr[ir] < arr[l]) {
        SWAP(arr[l],arr[ir])
        ISWAP(iarr[l],iarr[ir])
      }
      return;
    } else {
      mid=(l+ir) >> 1;
      SWAP(arr[mid],arr[l+1])
      ISWAP(iarr[mid],iarr[l+1])
      if (arr[l] > arr[ir]) {
        SWAP(arr[l],arr[ir])
        ISWAP(iarr[l],iarr[ir])
      }
      if (arr[l+1] > arr[ir]) {
        SWAP(arr[l+1],arr[ir])
        ISWAP(iarr[l+1],iarr[ir])
      }
      if (arr[l] > arr[l+1]) {
        SWAP(arr[l],arr[l+1])
        ISWAP(iarr[l],iarr[l+1])
      }
      i = l+1;
      j = ir;
      a = arr[l+1];
      ia = iarr[l+1];
      for (;;) {
        do i++; while (arr[i] < a);
        do j--; while (arr[j] > a);
        if (j < i) break;
        SWAP(arr[i],arr[j])
        ISWAP(iarr[i],iarr[j])
      }
      arr[l+1] = arr[j];
      arr[j] = a;
      iarr[l+1] = iarr[j];
      iarr[j] = ia;
      if (j >= k) ir = j-1;
      if (j <= k) l = i;
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeDiamondOrderAtom::memory_usage()
{
  double bytes = nmax * sizeof(double);
  bytes += maxneigh * sizeof(double); 
  bytes += maxneigh * sizeof(int); 

  return bytes;
}

/* ----------------------------------------------------------------------
   polar prefactor for spherical harmonic Y_l^m, where 
   Y_l^m (theta, phi) = prefactor(l, m, cos(theta)) * exp(i*m*phi)
------------------------------------------------------------------------- */

double ComputeDiamondOrderAtom::
polar_prefactor(int l, int m, double costheta) {
  const int mabs = abs(m);

  double prefactor = 1.0;
  for (int i=l-mabs+1; i < l+mabs+1; ++i)
    prefactor *= static_cast<double>(i);

  prefactor = sqrt(static_cast<double>(2*l+1)/(MY_4PI*prefactor))
    * associated_legendre(l,mabs,costheta);

  if ((m < 0) && (m % 2)) prefactor = -prefactor;
  //if (mabs%2) prefactor=-prefactor;

  return prefactor;
}

/* ----------------------------------------------------------------------
   associated legendre polynomial
------------------------------------------------------------------------- */

double ComputeDiamondOrderAtom::
associated_legendre(int l, int m, double x) {
  if (l < m) return 0.0;

  double p(1.0), pm1(0.0), pm2(0.0);

  if (m != 0) {
    const double sqx = sqrt(1.0-x*x);
    for (int i=1; i < m+1; ++i)
      p *= static_cast<double>(2*i-1) * sqx;
  }

  for (int i=m+1; i < l+1; ++i) {
    pm2 = pm1;
    pm1 = p;
    p = (static_cast<double>(2*i-1)*x*pm1
         - static_cast<double>(i+m-1)*pm2) / static_cast<double>(i-m);
  }

  return p;
}
double ComputeDiamondOrderAtom::smearing(double r) const {
    if (rsoft==0) {
        return 1;
    }
    else {
        return 1/(1+pow(r/rsoft,8));
    }
}

inline bool withinDev(double x0,double y0,double z0,double x,double y,double z,double devSq) {
    double xp=y0*z-z0*y;
    double yp=z0*x-x0*z;
    double zp=x0*y-y0*x;
    return (x0*x0+y0*y0+z0*z0)*devSq>=xp*xp+yp*yp+zp*zp;
}
bool ComputeDiamondOrderAtom::hydrogenBond(double x1,double y1,double z1,double x21,double y21,double z21,int *iH) const {
    if (devSq<0) {
        return true;
    }
    int s=0;
    int i;
    for (i=0;i<4;i++) {
        double *p=atom->x[iH[i]];
        s+=withinDev(x21,y21,z21,p[0]-x1,p[1]-y1,p[2]-z1,devSq);
    }
    return s==1;
}

