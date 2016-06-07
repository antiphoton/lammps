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
class FfsBranch {
public:
    FfsBranch() {
        if (!commInited) {
            size=world->size/local->size;
            initComm();
            commInited=true;
        }
    }
    ~FfsBranch() {
    }
private:
    static bool commInited;
    static void initComm() {
        MPI_Comm_dup(local->comm,&commLocal);
        MPI_Comm_split(world->comm,local->isLeader?0:MPI_UNDEFINED,world->rank,&commLeader);
    };
protected:
    static int size;
    static MPI_Comm commLeader,commLocal;
    static const int TAG_COUNTDOWN_DONE=1;
    static const int TAG_COUNTDOWN_TERMINATE=2;
    static const int TAG_FILEWRITER_LINE=3;
};
bool FfsBranch::commInited=false;
MPI_Comm FfsBranch::commLeader,FfsBranch::commLocal;
int FfsBranch::size=0;
struct FfsFileReader: public FfsBranch {
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
    };
    int getInt(const std::string &name) const {
        int y=0;
        if (world->isLeader) {
            const std::string v=getString(name);
            sscanf(v.c_str(),"%d",&y);
        }
        if (local->isLeader) {
            MPI_Bcast(&y,1,MPI_INT,0,commLeader);
        }
        MPI_Bcast(&y,1,MPI_INT,0,commLocal);
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
class FfsFileWriter: public FfsBranch {
protected:
    FfsFileWriter(const char *filename) {
        if (world->isLeader) {
            f=fopen(filename,"w");
        }
    }
    ~FfsFileWriter() {
        check();
        if (world->isLeader) {
            fclose(f);
        }
    }
    void writeln(const char *format,...) {
        if (!local->isLeader) {
            return ;
        }
        static char buffer[MAX_LENGTH+1];
        va_list args;
        va_start(args,format);
        vsprintf(buffer,format,args);
        va_end(args);
        putstr(buffer);
    }
    void check() {
        if (!world->isLeader) {
            return ;
        }
        static char buffer[MAX_LENGTH];
        while (1) {
            int flag;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE,TAG_FILEWRITER_LINE,commLeader,&flag,&status);
            if (flag) {
                int l;
                MPI_Get_count(&status,MPI_CHAR,&l);
                MPI_Recv(buffer,l,MPI_CHAR,status.MPI_SOURCE,status.MPI_TAG,commLeader,&status);
                putstr0(status.MPI_SOURCE,buffer);
            }
            else {
                break;
            }
        }
    }
private:
    FILE *f;
    void putstr(char *s) {
        s[MAX_LENGTH-1]='\0';
        if (world->isLeader) {
            putstr0(0,s);
        }
        else {
            int l=strlen(s);
            MPI_Send(s,l+1,MPI_CHAR,0,TAG_FILEWRITER_LINE,commLeader);
        }
    }
    void putstr0(int sender,const char *s) {
        static char buffer[MAX_LENGTH];
        sprintf(buffer,"%4d    %s",sender,s);
        fprintf(f,"%s\n",buffer);
        printf("%s\n",buffer);
    }
    static const int MAX_LENGTH=100;
};
class FfsTrajectoryWriter: public FfsFileWriter {
public:
    FfsTrajectoryWriter():FfsFileWriter("trajectory.txt") {
    }
    void check() {
        FfsFileWriter::check();
    }
    void writeln(const char *xyzInit,int lambdaInit,int velocitySeed,int64_t timestep,const char *xyzFinal,int lambdaFinal) {
        if (xyzInit==0) {
            FfsFileWriter::writeln("    (          )  >==%010d %20lld==>  %3d (xyz.%s)",velocitySeed,timestep,lambdaFinal,xyzFinal);
        }
        else {
            FfsFileWriter::writeln("%3d (xyz.%s)  >==%010d %20lld==>  %3d (xyz.%s)",lambdaInit,xyzInit,velocitySeed,timestep,lambdaFinal,xyzFinal);
        }
    }
};
class FfsCountdown: public FfsBranch {
public:
    FfsCountdown(int n) {
        remains=n;
        terminated=false;
    }
    ~FfsCountdown() {
    }
    void done(int n=1) {
        if (!local->isLeader) {
            return ;
        }
        if (terminated) {
            return ;
        }
        int x=n;
        if (world->isLeader) {
            remains-=x;
        }
        else {
            MPI_Send(&x,1,MPI_INT,0,TAG_COUNTDOWN_DONE,commLeader);
        }
    }
    bool next() {
        if (terminated) {
            return false;
        }
        int ret;
        if (local->isLeader) {
            if (world->isLeader) {
                while (1) {
                    int flag;
                    MPI_Status status;
                    MPI_Iprobe(MPI_ANY_SOURCE,TAG_COUNTDOWN_DONE,commLeader,&flag,&status);
                    if (flag) {
                        int x;
                        MPI_Recv(&x,1,MPI_INT,status.MPI_SOURCE,status.MPI_TAG,commLeader,&status);
                        remains-=x;
                    }
                    else {
                        break;
                    }
                }
                if (remains<=0) {
                    terminated=true;
                    int i;
                    for (i=1;i<size;i++) {
                        MPI_Send(0,0,MPI_INT,i,TAG_COUNTDOWN_TERMINATE,commLeader);
                    }
                    ret=0;
                }
                else {
                    ret=1;
                }
            }
            else {
                int flag;
                MPI_Status status;
                MPI_Iprobe(0,TAG_COUNTDOWN_TERMINATE,commLeader,&flag,&status);
                if (flag) {
                    terminated=true;
                    MPI_Recv(0,0,MPI_INT,0,TAG_COUNTDOWN_TERMINATE,commLeader,&status);
                    ret=0;
                }
                else {
                    ret=1;
                }
            }
        }
        MPI_Bcast(&ret,1,MPI_INT,0,commLocal);
        if (ret==0) {
            terminated=true;
        }
        return ret;
    }
private:
    bool terminated;
    int remains;
};
class FfsFileTree: public FfsBranch {
public:
    FfsFileTree(int layer):layer(layer) {
        a=new int[size];
        b=new int[size];
        int i;
        for (i=0;i<size;i++) {
            a[i]=0;
            b[i]=0;
        }
    }
    ~FfsFileTree() {
        delete[] b;
        delete[] a;
    }
    const std::string add() {
        int x;
        if (local->isLeader) {
            x=a[local->id];
            a[local->id]++;
        }
        MPI_Bcast(&x,1,MPI_INT,0,commLocal);
        return generateName(x);
    }
    void commit() {
        if (local->isLeader) {
            MPI_Allreduce(a,b,size,MPI_INT,MPI_SUM,commLeader);
        }
        MPI_Bcast(b,size,MPI_INT,0,commLocal);
        total=0;
        for (int i=0;i<size;i++) {
            total+=b[i];
        }
    }
    const std::string get(int x) const {
        x%=total;
        int i;
        for (i=0;i<size;i++) {
            if (x<b[i]) {
                return generateName(x,i);
            }
            else {
                x-=b[i];
            }
        }
    }
private:
    int layer;
    int *a,*b;
    int total;
    const std::string generateName(int x, int branchId=-1) const {
        if (branchId==-1) {
            branchId=local->id;
        }
        static char c[100];
        sprintf(c,"%d__%d_%d",layer,branchId,x);
        return c;
    }
};
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
class FfsRandomGenerator: public FfsBranch {
    public:
        FfsRandomGenerator() {
            if (!local->isLeader) {
                return ;
            }
            std::srand(std::time(0)+world->rank);
            std::rand();
            std::rand();
        }
        int get() {
            int x;
            if (local->isLeader) {
                x=std::rand();
            }
            MPI_Bcast(&x,1,MPI_INT,0,commLocal);
            return x;
        }
};

int createVelocity(LAMMPS *lammps,int temp,FfsRandomGenerator *pRng) {
    static char str[100];
    int seed=pRng->get();
    sprintf(str,"velocity all create %d %d",temp,seed);
    lammps_command(lammps,str);
    return seed;
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
    runBatch(0);
    LAMMPS *lammps=new LAMMPS(argc,argv,local->comm);
    lammps->input->file();
    FfsTrajectoryWriter fileTrajectory;
    int temperatureMean=ffsParams->getInt("temperature");
    static int lambda_A=ffsParams->getInt("lambda_A");
    FfsRandomGenerator rng;
    FfsFileTree *lastTree,*currentTree;
    if (1) {
        int velocitySeed=createVelocity(lammps,temperatureMean,&rng);
        FfsCountdown *fcd=new FfsCountdown(ffsParams->getInt("config_each_lambda"));
        lammps_command(lammps,(char *)"run 0 pre yes post no");
        lastTree=0;
        currentTree=new FfsFileTree(0);
        while (1) {
            bool ready=false;
            int lambda;
            while (1) {
                runBatch(lammps);
                const double *lambdaReuslt=(const double *)lammps_extract_compute(lammps,(char *)"lambda",0,1);
                lambda=(int)lambdaReuslt[0];
                static int lambda_0=ffsParams->getInt("lambda_0");
                if (lambda<=lambda_A) {
                    ready=true;
                }
                if (ready&&lambda>=lambda_0) {
                    ready=false;
                    break;
                }
                fileTrajectory.check();
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
            const std::string xyzFinal=currentTree->add();
            sprintf(strDump,"write_dump all xyz pool/xyz.%s",xyzFinal.c_str());
            fileTrajectory.writeln((const char *)0,0,velocitySeed,timestep,xyzFinal.c_str(),lambda);
            lammps_command(lammps,strDump);
            fcd->done();
        }
        lammps_command(lammps,(char *)"run 0 pre no post yes");
    }
    const int n=5;
    int lambda[n]={11,19,22,25,28};
    for (int i=1;i+1<n;i++) {
        delete lastTree;
        lastTree=currentTree;
        currentTree=new FfsFileTree(i);
        lastTree->commit();
        FfsCountdown *fcd=new FfsCountdown(ffsParams->getInt("config_each_lambda"));
        while (1) {
            static char strReadData[100];
            const std::string xyzInit=lastTree->get(rng.get());
            sprintf(strReadData,"read_dump pool/xyz.%s 0 x y z box no format xyz",xyzInit.c_str());
            lammps_command(lammps,strReadData);
            int velocitySeed=createVelocity(lammps,temperatureMean,&rng);
            lammps_command(lammps,(char *)"run 0 pre yes post no");
            int lambda_calc,lambda_next=lambda[i+1];
            while (1) {
                runBatch(lammps);
                const double *lambdaReuslt=(const double *)lammps_extract_compute(lammps,(char *)"lambda",0,1);
                lambda_calc=(int)lambdaReuslt[0];
                if (lambda_calc<=lambda_A||lambda_calc>=lambda_next) {
                    break;
                }
                fileTrajectory.check();
                if (!fcd->next()) {
                    break;
                }
            }
            if (!fcd->next()) {
                delete fcd;
                break;
            }
            int64_t timestep=lammps->update->ntimestep;
            lammps_command(lammps,(char *)"run 0 pre no post yes");
            if (lambda_calc<=lambda_A) {
                continue;
            }
            if (lambda_calc>=lambda_next) {
                static char strDump[100];
                const std::string xyzFinal=currentTree->add();
                sprintf(strDump,"write_dump all xyz pool/xyz.%s",xyzFinal.c_str());
                fileTrajectory.writeln(xyzInit.c_str(),-1,velocitySeed,timestep,xyzFinal.c_str(),lambda_calc);
                lammps_command(lammps,strDump);
                fcd->done();
                continue;
            }
        }
    }
    delete lammps;
    delete local;
    delete world;
    return 0;
}

