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
#include "compute_biggest.h"
#include "update.h"
#include "modify.h"
#include "atom.h"
#include "group.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define INVOKED_VECTOR 2
#define INVOKED_ARRAY 4
#define INVOKED_PERATOM 8
#define INVOKED_LOCAL 16

/* ---------------------------------------------------------------------- */

ComputeBiggest::ComputeBiggest(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 4) {
      error->all(FLERR,"Illegal compute biggest command");
  }

  int iarg=3;
  if (strncmp(arg[iarg],"c_",2) == 0) {
      int iCompute=modify->find_compute(&arg[iarg][2]);
      if (iCompute<0) {
          error->all(FLERR,"Illegal compute nuclei/atom command");
      }
      compute_nuclei=modify->compute[iCompute];
      if (compute_nuclei->peratom_flag!=1) {
          error->all(FLERR,"Illegal compute nuclei/atom command");
      }
  }
  else {
      error->all(FLERR,"Illegal compute nuclei/atom command");
  }

  vector_flag = 1;
  size_vector = 2;
  extscalar = 0;
  extvector = 0;

  vector = new double[2];
}

/* ---------------------------------------------------------------------- */

ComputeBiggest::~ComputeBiggest()
{
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeBiggest::init()
{
}

/* ----------------------------------------------------------------------
   compute the radius-of-biggest tensor of group of atoms
   around center-of-mass cm
   must unwrap atoms to compute Rg tensor correctly
------------------------------------------------------------------------- */

void ComputeBiggest::compute_vector()
{
  invoked_vector = update->ntimestep;

  {
      if (!(compute_nuclei->invoked_flag & INVOKED_PERATOM)) {
          compute_nuclei->compute_peratom();
          compute_nuclei->invoked_flag |= INVOKED_PERATOM;
      }
  }

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;

  double dx,dy,dz,massone;
  double unwrap[3];

  double rg[6];
  rg[0] = rg[1] = rg[2] = rg[3] = rg[4] = rg[5] = 0.0;

  int maxTagLocal=0;

  for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (tag[i]>maxTagLocal) {
          maxTagLocal=tag[i];
      }
  }
  int maxTag;
  MPI_Allreduce(&maxTagLocal,&maxTag,1,MPI_INT,MPI_MAX,world);
  int *countLocal,*countGlobal;
  memory->create(countLocal,1+maxTag,"biggest:countLocal");
  memory->create(countGlobal,1+maxTag,"biggest:countLocal");
  for (int i=0;i<=maxTag;i++) {
      countLocal[i]=0;
  }

  double *vector_nuclei=compute_nuclei->vector_atom;
  for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      countLocal[(int)(vector_nuclei[i])]++;
  }
  MPI_Allreduce(countLocal,countGlobal,1+maxTag,MPI_INT,MPI_SUM,world);

  int maxCount=-1,iMax;
  for (int i = 1; i <= maxTag; i++) {
      int current=countGlobal[i];
      if (current>maxCount) {
          maxCount=current;
          iMax=i;
      }
  }

  vector[0]=maxCount;
  vector[1]=iMax;

  memory->destroy(countGlobal);
  memory->destroy(countLocal);
}
