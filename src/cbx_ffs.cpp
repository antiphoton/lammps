#include<stdio.h>
#include<string.h>
#include<mpi.h>
#include<cstdlib>
#include<ctime>
#include<map>
#include<string>
#include"lammps.h"
#include"library.h"
#include"input.h"
#include"update.h"
#include"cbx_ffs.h"
using namespace LAMMPS_NS;
struct MpiInfo {
    const int LEADER;
    const int id;
    MPI_Comm comm;
    int size,rank;
    bool isLeader;
    MpiInfo(MPI_Comm comm,int id=0):comm(comm),LEADER(0),id(id) {
        MPI_Comm_size(comm,&size);
        MPI_Comm_rank(comm,&rank);
        isLeader=LEADER==rank;
    };
};
const MpiInfo *world,*local;
struct FfsFileReader {
    std::map<std::string,std::string> dict;
    FfsFileReader(const char *filename) {
        if (world->isLeader) {
            FILE *f=fopen(filename,"r");
            static char cLine[1024];
            while (fgets(cLine,sizeof(cLine),f)) {
                std::string sLine(cLine);
                int l=sLine.find_first_of('#');
                if (l!=std::string::npos) {
                    sLine=sLine.substr(0,l);
                }
                const std::string SPACE=" \t\n\v\f\r";
                int begin1=sLine.find_first_not_of(SPACE);
                if (begin1==std::string::npos) {
                    continue;
                }
                int end1=sLine.find_first_of(SPACE,begin1);
                if (end1==std::string::npos) {
                    end1=sLine.length();
                }
                std::string key=sLine.substr(begin1,end1-begin1);
                dict[key]="";
                int begin2=sLine.find_first_not_of(SPACE,end1);
                if (begin2==std::string::npos) {
                    continue;
                }
                int end2=sLine.find_first_of(SPACE,begin2);
                if (end2==std::string::npos) {
                    end2=sLine.length();
                }
                std::string value=sLine.substr(begin2,end2-begin2);
                dict[key]=value;
            }
            fclose(f);
        }
        MPI_Barrier(world->comm);
    };
    int getInt(const std::string &name) const {
        const std::string v=getString(name);
        int y=0;
        if (world->isLeader) {
            sscanf(v.c_str(),"%d",&y);
        }
        MPI_Bcast(&y,1,MPI_INT,world->LEADER,world->comm);
        return y;
    }
private:
    const std::string getString(const std::string &name) const {
        std::map<std::string,std::string>::const_iterator i=dict.find(name);
        if (i==dict.end()) {
            return "";
        }
        else {
            return i->second;
        }
    }
};
const FfsFileReader *ffsParams;

bool ffsRequested(int argc, char **argv) {
    int i;
    for (i=0;i<argc;i++) {
        if (strcmp(argv[i],"-ffs")==0) {
            if (i+2<argc) {
                int numberShooting;
                if (sscanf(argv[i+1],"%d",&numberShooting)==1) {
                    MPI_Comm worldComm;
                    MPI_Comm_dup(MPI_COMM_WORLD,&worldComm);
                    world=new MpiInfo(worldComm);
                    if (world->size>0&&numberShooting>0&&world->size%numberShooting==0) {
                        int cpuEach=world->size/numberShooting;
                        int myShooting=world->rank/cpuEach;
                        MPI_Comm localComm;
                        MPI_Comm_split(world->comm,myShooting,world->rank,&localComm);
                        local=new MpiInfo(localComm,myShooting);
                    }
                }
                else {
                    return false;
                }
                ffsParams=new FfsFileReader(argv[i+2]);
                return true;
            }
        }
    }
    return false;
}
int createSharedRandomSeed() {
    MPI_Barrier(local->comm);
    int x;
    if (local->isLeader) {
        static bool inited=false;
        if (!inited) {
            std::srand(std::time(0)+world->rank);
            inited=true;
        }
        x=std::rand();
    }
    MPI_Bcast(&x,1,MPI_INT,local->LEADER,local->comm);
    return x;
}
void createVelocity(LAMMPS *lammps) {
    static char str[100];
    static bool inited=false;
    if (!inited) {
        int temperatureMean=ffsParams->getInt("temperature");
        int velocitySeed=createSharedRandomSeed();
        sprintf(str,"velocity all create %d %d",temperatureMean,velocitySeed);
        inited=true;
    }
    if (lammps) {
        lammps_command(lammps,str);
    }
};
void runBatch(LAMMPS *lammps) {
    static char str[100];
    static bool inited=false;
    if (!inited) {
        int everyStep=ffsParams->getInt("check_every");
        sprintf(str,"run %d pre no post no",everyStep);
        inited=true;
    }
    if (lammps) {
        lammps_command(lammps,str);
    }
}
const int lammps_every_step=20;
const int target_pool_count=2;
const int lambda_A=11;
const int lambda_0=21;
int ffs_main(int argc, char **argv) {
    createVelocity(0);
    runBatch(0);
    LAMMPS *lammps=new LAMMPS(argc,argv,local->comm);
    lammps->input->file();
    createVelocity(lammps);
    int current_pool_found=0;
    lammps_command(lammps,(char *)"run 0 pre yes post no");
    while (current_pool_found<target_pool_count) {
        bool ready=false;
        const double *lambdaReuslt;
        while (1) {
            runBatch(lammps);
            lambdaReuslt=(const double *)lammps_extract_compute(lammps,(char *)"lambda",0,1);
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
        sprintf(strDump,"write_dump all custom pool/%lld-%d.lammpstraj id x y z",timestep,local->id);
        printf("%d\t%lld\n",local->id,timestep);
        lammps_command(lammps,strDump);
        current_pool_found++;
    }
    lammps_command(lammps,(char *)"run 0 pre no post yes");
    delete lammps;
    delete local;
    delete world;
    return 0;
}

