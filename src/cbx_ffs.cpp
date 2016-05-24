#include<stdio.h>
#include<string.h>
#include<mpi.h>
#include<cstdlib>
#include<ctime>
#include"lammps.h"
#include"library.h"
#include"input.h"
#include"update.h"
#include"cbx_ffs.h"
using namespace LAMMPS_NS;
static int worldSize,worldRank;
static int numberShooting;
static int cpuEach;
static int myShooting;
static MPI_Comm myComm;
static int localRank;
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
int createSharedRandomSeed() {
    const static int LEADER=0;
    MPI_Barrier(myComm);
    int x;
    if (localRank==LEADER) {
        static bool inited=false;
        if (!inited) {
            std::srand(std::time(0)+worldRank);
            inited=true;
        }
        x=std::rand();
    }
    MPI_Bcast(&x,1,MPI_INT,LEADER,myComm);
    return x;
}
const int lammps_every_step=20;
const int target_pool_count=2;
const int lambda_A=11;
const int lambda_0=21;
int ffs_main(int argc, char **argv) {
    MPI_Comm_split(MPI_COMM_WORLD,myShooting,worldRank,&myComm);
    MPI_Comm_rank(myComm,&localRank);
    LAMMPS *lammps=new LAMMPS(argc,argv,myComm);
    lammps->input->file();
    char *strVelocityCreate=new char[100];
    int velocitySeed=createSharedRandomSeed();
    sprintf(strVelocityCreate,"velocity all create 220 %d",velocitySeed);
    lammps_command(lammps,strVelocityCreate);
    delete[] strVelocityCreate;
    int current_pool_found=0;
    lammps_command(lammps,"run 0 pre yes post no");
    while (current_pool_found<target_pool_count) {
        bool ready=false;
        const double *lambdaReuslt;
        while (1) {
            lammps_command(lammps,"run 20 pre no post no");
            lambdaReuslt=(const double *)lammps_extract_compute(lammps,"lambda",0,1);
            int lambda=(int)lambdaReuslt[0];
            if (lambda<=lambda_A) {
                ready=true;
            }
            if (ready&&lambda>=lambda_0) {
                ready=false;
                break;
            }
        }
        int64_t timestep=lammps->update->ntimestep;
        static char strDump[100];
        sprintf(strDump,"write_dump all custom pool/%lld-%d.lammpstraj id x y z",timestep,myShooting);
        printf("%d\t%lld\n",myShooting,timestep);
        lammps_command(lammps,strDump);
        current_pool_found++;
    }
    lammps_command(lammps,"run 0 pre no post yes");
    delete lammps;
    return 0;
}

