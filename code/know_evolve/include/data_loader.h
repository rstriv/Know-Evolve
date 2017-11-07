#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "config.h"
#include "imatrix.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "dataset.h"

class DataLoader
{
public:

    static void Init();

    static void Reset();

    static SparseMat<mode, Dtype>* LoadOneHot(int dim, int id);
    
    static std::pair<SparseMat<mode, Dtype>*, SparseMat<mode, Dtype>* > LoadActFeat(Event* e);
    static std::pair<DenseMat<mode, Dtype>*, DenseMat<mode, Dtype>* > LoadDurFeat(Event* e);
    static DenseMat<mode, Dtype>* LoadLabel(Dtype dur);

    static DenseMat<mode, Dtype>* GetDenseMat();
    static SparseMat<mode, Dtype>* GetSparseMat();

    static std::vector< DenseMat<mode, Dtype>* > dense_mats;
    static std::vector< SparseMat<mode, Dtype>* > sparse_mats;
    static int dense_used, sparse_used;
};

#endif
