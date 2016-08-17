/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(nuclei/atom,ComputeNucleiAtom)

#else

#ifndef LMP_COMPUTE_NUCLEI_ATOM_H
#define LMP_COMPUTE_NUCLEI_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeNucleiAtom : public Compute {
 public:
  ComputeNucleiAtom(class LAMMPS *, int, char **);
  ~ComputeNucleiAtom();
  void init();
  void init_list(int, class NeighList *);
  void compute_peratom();
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  double memory_usage();

 private:
  int nConditions;
  int *value2index;
  bool *compareDirection;// true for greater than, false for less than
  double *threshold;
  double hardNeighbourDistance;
  int hardNeighbourCount;
  int oxygenId;

  int nmax,commflag;
  double cutsq;
  class NeighList *list;
  double *nucleiID;
  double *isSolid;
  void invokePrerequisites() const;
  bool checkSolid(int m) const;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot use compute nuclei/atom unless atoms have IDs

Atom IDs are used to identify nuclei.

E: Compute nuclei/atom requires a pair style be defined

This is so that the pair style defines a cutoff distance which
is used to find nuclei.

E: Compute nuclei/atom cutoff is longer than pairwise cutoff

Cannot identify nuclei beyond cutoff.

W: More than one compute nuclei/atom

It is not efficient to use compute nuclei/atom  more than once.

*/
