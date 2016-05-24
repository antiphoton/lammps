#include<stdarg.h>
#include<stdio.h>
#include<string.h>
#include<mpi.h>
#include<cstdlib>
#include<ctime>
#include<map>
#include<queue>
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
class FfsFileWriter {
public:
    FfsFileWriter(const char *filename) {
        if (local->isLeader) {
            static char str[1024];
            sprintf(str,"%s-%03d",filename,local->id);
            f=fopen(str,"w");
        }
    }
    ~FfsFileWriter() {
        if (local->isLeader) {
            fclose(f);
        }
    }
    int write(const char *format,...) {
        int ret=0;
        if (local->isLeader) {
            static char buffer[1024];
            va_list args;
            va_start(args,format);
            ret=vsprintf(buffer,format,args);
            va_end(args);
            printf("%s",buffer);
            fprintf(f,"%s",buffer);
        }
        return ret;
    }
private:
    FILE *f;
};
class FfsCountdown {
public:
    FfsCountdown(int n) {
        if (!inited) {
            initComm();
            inited=true;
        }
        if (local->isLeader) {
            if (world->isLeader) {
            }
            else {
                std::pair<MPI_Request,int*> p;
                p.second=new int;
                *p.second=0;
                MPI_Irecv(0,0,MPI_INT,world->LEADER,1,commWorld,&p.first);
                q2.push(p);
            }
        }
        broadcasted=false;
        remains=n;
    }
    ~FfsCountdown() {
        MPI_Barrier(commWorld);
        releaseQueue(q1,true);
        releaseQueue(q2,true);
    }
    void done(int n=1) {
        if (local->isLeader) {
            if (world->isLeader) {
                remains-=n;
                broadcastTermination();
            }
            else {
                std::pair<MPI_Request,int*> p;
                p.second=new int;
                *p.second=n;
                MPI_Isend(p.second,1,MPI_INT,world->LEADER,0,commWorld,&p.first);
                q1.push(p);
            }
        }
    }
    bool next() {
        if (remains<=0) {
            return false;
        }
        if (local->isLeader) {
            if (world->isLeader) {
                if (remains>0) {
                    if (q1.empty()) {
                        std::pair<MPI_Request,int*> p;
                        p.second=new int;
                        *p.second=0;
                        MPI_Irecv(p.second,1,MPI_INT,MPI_ANY_SOURCE,0,commWorld,&p.first);
                        q1.push(p);
                    }
                    while (!q1.empty()) {
                        int flag;
                        MPI_Status status;
                        releaseQueue(q1,false,&flag,&status);
                        if (!flag) {
                            break;
                        }
                        remains-=*q1.front().second;
                        delete q1.front().second;
                        q1.pop();
                    }
                    broadcastTermination();
                }
            }
            else {
                int flag;
                MPI_Status status;
                releaseQueue(q2,false,&flag,&status);
                if (flag) {
                    remains=-1;
                    delete q2.front().second;
                    q2.pop();
                }
            }
        }
        MPI_Bcast(&remains,1,MPI_INT,local->LEADER,commLocal);
        return remains>0;
    }
private:
    static bool inited;
    static MPI_Comm commWorld,commLocal;
    static void initComm() {
        MPI_Comm_dup(world->comm,&commWorld);
        MPI_Comm_dup(local->comm,&commLocal);
    };
    bool broadcasted;
    void broadcastTermination() {
        if (!world->isLeader) {
            return ;
        }
        if (remains>0) {
            return ;
        }
        if (broadcasted) {
            return ;
        }
        broadcasted=true;
        int i;
        for (i=1;i<world->size;i++) {
            std::pair<MPI_Request,int*> p;
            p.second=new int;
            *p.second=0;
            MPI_Isend(0,0,MPI_INT,i,1,commWorld,&p.first);
            q2.push(p);
        }
        MPI_Bcast(&remains,1,MPI_INT,local->LEADER,commLocal);
    };
    void releaseQueue(std::queue< std::pair<MPI_Request,int*> > &q,bool hard,int *flag=0,MPI_Status *status=0) {
        while (!q.empty()) {
            std::pair<MPI_Request,int*> f=q.front();
            if (f.first!=MPI_REQUEST_NULL) {
                if (hard) {
                    MPI_Cancel(&f.first);
                }
                else {
                    MPI_Test(&f.first,flag,status);
                    break;
                }
            }
            delete f.second;
            q.pop();
        }
    };
    std::queue< std::pair<MPI_Request,int*> > q1,q2;
    int remains;
};
bool FfsCountdown::inited=false;
MPI_Comm FfsCountdown::commWorld,FfsCountdown::commLocal;
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
int ffs_main(int argc, char **argv) {
    createVelocity(0);
    runBatch(0);
    LAMMPS *lammps=new LAMMPS(argc,argv,local->comm);
    lammps->input->file();
    FfsFileWriter fileSummary("total_time.dat");
    createVelocity(lammps);
    FfsCountdown *fcd=new FfsCountdown(ffsParams->getInt("config_each_lambda"));
    lammps_command(lammps,(char *)"run 0 pre yes post no");
    while (1) {
        bool ready=false;
        const double *lambdaReuslt;
        while (1) {
            runBatch(lammps);
            lambdaReuslt=(const double *)lammps_extract_compute(lammps,(char *)"lambda",0,1);
            int lambda=(int)lambdaReuslt[0];
            static int lambda_A=ffsParams->getInt("lambda_A");
            static int lambda_0=ffsParams->getInt("lambda_0");
            if (lambda<=lambda_A) {
                ready=true;
            }
            if (ready&&lambda>=lambda_0) {
                ready=false;
                break;
            }
            if (!fcd->next()) {
                break;
            }
        }
        if (!fcd->next()) {
            delete fcd;
            break;
        }
        int64_t timestep=lammps->update->ntimestep;
        static char strDump[100];
        sprintf(strDump,"write_dump all custom pool/%lld-%d.lammpstraj id x y z",timestep,local->id);
        fileSummary.write("%d\t%lld\n",local->id,timestep);
        lammps_command(lammps,strDump);
        fcd->done();
    }
    lammps_command(lammps,(char *)"run 0 pre no post yes");
    delete lammps;
    delete local;
    delete world;
    return 0;
}

