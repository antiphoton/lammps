#include<stdio.h>
#include<string.h>
#include<mpi.h>
#include"lammps.h"
#include"library.h"
#include"input.h"
#include"cbx_ffs.h"
using namespace LAMMPS_NS;
static int worldSize,worldRank;
static int numberShooting;
static int cpuEach;
static int myShooting;
static MPI_Comm myComm;
bool ffsRequested(int argc, char **argv) {
    int i;
    for (i=0;i<argc;i++) {
        if (strcmp(argv[i],"-ffs")==0) {
            if (i+1<argc) {
                int numberShooting;
                if (sscanf(argv[i+1],"%d",&numberShooting)==1) {
                    MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
                    MPI_Comm_rank(MPI_COMM_WORLD,&worldRank);
                    if (worldSize>0&&numberShooting>0&&worldSize%numberShooting==0) {
                        cpuEach=worldSize/numberShooting;
                        myShooting=worldRank/cpuEach;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}
int ffs_main(int argc, char **argv) {
    MPI_Comm_split(MPI_COMM_WORLD,myShooting,worldRank,&myComm);
    LAMMPS *lammps=new LAMMPS(argc,argv,myComm);
    //lammps_command(lammps,"variable s equal 10");
    //lammps->input->file("in.ffs");
    int n=lammps_get_natoms(lammps);
    delete lammps;
    return 0;
}

