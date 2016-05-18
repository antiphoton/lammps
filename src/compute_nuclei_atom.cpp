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

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "compute_nuclei_atom.h"
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

#include "group.h"

using namespace LAMMPS_NS;

enum{X,V,F,COMPUTE,FIX,VARIABLE};

/* ---------------------------------------------------------------------- */

ComputeNucleiAtom::ComputeNucleiAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal compute nuclei/atom command");

  if ((narg-4)%3!=0) error->all(FLERR,"Illegal compute nuclei/atom command");

  nConditions=(narg-4)/3;
  which=new int[nConditions];
  value2index=new int[nConditions];
  compareDirection=new bool[nConditions];
  threshold=new double[nConditions];
  hardNeighbourDistance=0;
  hardNeighbourCount=0;
  int iCondition=0;
  int iarg = 4;
  while (iarg < narg) {
    if (strncmp(arg[iarg],"c_",2) == 0 ||
        strncmp(arg[iarg],"f_",2) == 0 ||
        strncmp(arg[iarg],"v_",2) == 0) {
      if (arg[iarg][0] == 'c') {
          which[iCondition] = COMPUTE;
          value2index[iCondition]=modify->find_compute(&arg[iarg][2]);
      //} else if (arg[iarg][0] == 'f') {
      //    which[iCondition] = FIX;
      //}
      //else if (arg[iarg][0] == 'v') {
      //    which[iCondition] = VARIABLE;
      }
      else error->all(FLERR,"Illegal compute diamondorder/atom command");

      if (value2index[iCondition]<0) {
          error->all(FLERR,"Illegal compute diamondorder/atom command");
      }

      if (arg[iarg+1][0] == 'g') compareDirection[iCondition] = true;
      else if (arg[iarg+1][0] == 'l') compareDirection[iCondition] = false;
      else error->all(FLERR,"Illegal compute diamondorder/atom command");

      threshold[iCondition]=force->numeric(FLERR,arg[iarg+2]);
      iCondition++;
      iarg+=2;
    }
    else if (strcmp(arg[iarg],"hardNeighbour") == 0) {
        hardNeighbourDistance=force->numeric(FLERR,arg[iarg+1]);
        hardNeighbourCount=force->numeric(FLERR,arg[iarg+2]);
        iarg+=2;
    } else error->all(FLERR,"Illegal compute diamondorder/atom command");
    iarg++;
  }

  double cutoff = force->numeric(FLERR,arg[3]);
  cutsq = cutoff*cutoff;

  peratom_flag = 1;
  size_peratom_cols = 0;
  comm_forward = 2;

  nmax = 0;
  nucleiID = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeNucleiAtom::~ComputeNucleiAtom()
{
  memory->destroy(isSolid);
  memory->destroy(nucleiID);
  delete[] which;
  delete[] value2index;
  delete[] compareDirection;
  delete[] threshold;
}

/* ---------------------------------------------------------------------- */

void ComputeNucleiAtom::init()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Cannot use compute nuclei/atom unless atoms have IDs");
  if (force->pair == NULL)
    error->all(FLERR,"Compute nuclei/atom requires a pair style be defined");
  if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR,
               "Compute nuclei/atom cutoff is longer than pairwise cutoff");

  // need an occasional full neighbor list
  // full required so that pair of atoms on 2 procs both set their nucleiID

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"nuclei/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute nuclei/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeNucleiAtom::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeNucleiAtom::compute_peratom()
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;

  invoked_peratom = update->ntimestep;

  // grow nucleiID array if necessary

  if (atom->nlocal+atom->nghost > nmax) {
    memory->destroy(isSolid);
    memory->destroy(nucleiID);
    nmax = atom->nmax;
    memory->create(isSolid,nmax,"nuclei/atom:isSolid");
    memory->create(nucleiID,nmax,"nuclei/atom:nucleiID");
    vector_atom = nucleiID;
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // if group is dynamic, insure ghost atom masks are current

  if (group->dynamic[igroup]) {
    commflag = 0;
    comm->forward_comm_compute(this);
  }

  // every atom starts in its own nuclei, with nucleiID = atomID

  tagint *tag = atom->tag;
  int *mask = atom->mask;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
        nucleiID[i] = tag[i];
    }
    else {
        nucleiID[i] = 0;
    }
    isSolid[i]=checkSolid(i)?1:0;
  }

  // loop until no more changes on any proc:
  // acquire nucleiIDs of ghost atoms
  // loop over my atoms, checking distance to neighbors
  // if both atoms are in nuclei, assign lowest nucleiID to both
  // iterate until no changes in my atoms
  // then check if any proc made changes

  commflag = 1;
  double **x = atom->x;

  int change,done,anychange;

  while (1) {
    comm->forward_comm_compute(this);

    change = 0;
    while (1) {
      done = 1;
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (!(mask[i] & groupbit)) continue;
        if (!isSolid[i]) {
            continue;
        }

        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;
          if (!(mask[j] & groupbit)) continue;
          if (nucleiID[i] == nucleiID[j]) continue;
          if (!isSolid[j]) {
              continue;
          }

          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;
          if (rsq < cutsq) {
              int iMin = MIN(nucleiID[i],nucleiID[j]);
              nucleiID[i] = nucleiID[j] = iMin;
              done = 0;
          }
        }
      }
      if (!done) change = 1;
      if (done) break;
    }

    // stop if all procs are done

    MPI_Allreduce(&change,&anychange,1,MPI_INT,MPI_MAX,world);
    if (!anychange) break;
  }
  if (1) {
      comm->forward_comm_compute(this);
      for (ii = 0; ii < inum; ii++) {
          i = ilist[ii];
          if (!(mask[i] & groupbit)) continue;
          if (isSolid[i]) {
              continue;
          }
          nucleiID[i]=0;

          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
          jlist = firstneigh[i];
          jnum = numneigh[i];

          for (jj = 0; jj < jnum; jj++) {
              j = jlist[jj];
              j &= NEIGHMASK;
              if (!(mask[j] & groupbit)) continue;
              if (nucleiID[i] == nucleiID[j]) continue;
              if (!isSolid[j]) {
                  continue;
              }

              delx = xtmp - x[j][0];
              dely = ytmp - x[j][1];
              delz = ztmp - x[j][2];
              rsq = delx*delx + dely*dely + delz*delz;
              if (rsq < cutsq) {
                  if (nucleiID[i]==0||nucleiID[i]>nucleiID[j]) {
                      double iMin = nucleiID[j] +0.5;
                      nucleiID[i] = iMin;
                  }
              }
          }
      }
      MPI_Barrier(world);
  }
}

/* ---------------------------------------------------------------------- */

int ComputeNucleiAtom::pack_forward_comm(int n, int *list, double *buf,
                                          int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  if (commflag) {
          for (i = 0; i < n; i++) {
              j = list[i];
              buf[m++] = isSolid[j];
              buf[m++] = nucleiID[j];
          }
  } else {
    int *mask = atom->mask;
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = ubuf(mask[j]).d;
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeNucleiAtom::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  if (commflag)
        for (i = first; i < last; i++) {
            isSolid[i] = buf[m++];
            nucleiID[i] = buf[m++];
        }
  else {
    int *mask = atom->mask;
    for (i = first; i < last; i++) {
        mask[i] = (int) ubuf(buf[m++]).i;
        mask[i] = (int) ubuf(buf[m++]).i;
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeNucleiAtom::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}

bool ComputeNucleiAtom::checkSolid(int i) const {
    if (hardNeighbourDistance>0) {
        double disSqr=hardNeighbourDistance*hardNeighbourDistance;
        int neighbourCount=0;
        double **p=atom->x;
        double x1=p[i][0];
        double y1=p[i][1];
        double z1=p[i][2];
        int *jlist=list->firstneigh[i];
        int jnum=list->numneigh[i];
        int jj;
        for (jj=0;jj<jnum;jj++) {
            int j=jlist[jj];
            double x2=p[j][0];
            double y2=p[j][1];
            double z2=p[j][2];
            double xd=x2-x1;
            double yd=y2-y1;
            double zd=z2-z1;
            double currentDisSqr=xd*xd+yd*yd+zd*zd;
            if (currentDisSqr<=disSqr) {
                neighbourCount++;
            }
        }
        if (neighbourCount<hardNeighbourCount) {
            return false;
        }
    }
    bool ret=true;
    int m;
    for (m=0;m<nConditions;m++) {
        int wh=which[m];
        int vidx=value2index[m];
        bool cd=compareDirection[m];
        double td=threshold[m];
        if (wh==COMPUTE) {
            Compute *compute=modify->compute[vidx];
            double *comp_vec=compute->vector_atom;
            if (cd) {
                ret=ret&&comp_vec[i]>td;
            }
            else {
                ret=ret&&comp_vec[i]<td;
            }
        }
        if (ret==false) {
            break;
        }
    }
    return ret;
}

