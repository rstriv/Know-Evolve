#include "data_loader.h"
#include "dataset.h"
#include "config.h"

void DataLoader::Init()
{
	dense_mats.clear();
	sparse_mats.clear();
	dense_used = 0;
	sparse_used = 0;
}

void DataLoader::Reset()
{
	dense_used = 0;
	sparse_used = 0;
}

SparseMat<mode, Dtype>* DataLoader::LoadOneHot(int dim, int id)
{
	auto& mat = *(GetSparseMat());
	assert(id >= 0 && id < dim);
	mat.Resize(1, dim);
	mat.ResizeSp(1, 2);

	mat.data->ptr[0] = 0;
	mat.data->ptr[1] = 1;
	mat.data->val[0] = 1.0;
	mat.data->col_idx[0] = id;

	return &mat;
}

std::pair<SparseMat<mode, Dtype>*, SparseMat<mode, Dtype>* > DataLoader::LoadActFeat(Event* e)
{
	auto& subject_feat = *(GetSparseMat());
	auto& object_feat = *(GetSparseMat());

	subject_feat.Resize(1, cfg::num_feats);
	subject_feat.ResizeSp(e->feat.size(), 2);
	subject_feat.data->ptr[0] = 0;
	subject_feat.data->ptr[1] = e->feat.size();

	for (size_t i = 0; i < e->feat.size(); ++i)
	{
		assert(e->feat[i].first >= 0 && e->feat[i].first < cfg::num_feats);
		subject_feat.data->col_idx[i] = e->feat[i].first;
		subject_feat.data->val[i] = e->feat[i].second;
	}
	object_feat.CopyFrom(subject_feat);
	return std::pair<SparseMat<mode, Dtype>*, SparseMat<mode, Dtype>* >(&subject_feat, &object_feat);
}

std::pair<DenseMat<mode, Dtype>*, DenseMat<mode, Dtype>* > DataLoader::LoadDurFeat(Event* e)
{
	auto& subject_dur = *(GetDenseMat());
	auto& object_dur = *(GetDenseMat());

	subject_dur.Resize(1, 1);
	object_dur.Resize(1, 1);

	Dtype s_dur = (e->subj_prev_event == nullptr) ? 0.0 : e->t - e->subj_prev_event->t;

	subject_dur.data[0] = s_dur;


	Dtype o_dur = (e->obj_prev_event == nullptr) ? 0.0 : e->t - e->obj_prev_event->t;

	object_dur.data[0] = o_dur;

	return std::pair<DenseMat<mode, Dtype>*, DenseMat<mode, Dtype>* >(&subject_dur, &object_dur);
}

DenseMat<mode, Dtype>* DataLoader::LoadLabel(Dtype dur)
{
	auto& time_label = *(GetDenseMat());
    	time_label.Resize(1, 1);

	time_label.data[0] = dur;

	return &time_label;
}

DenseMat<mode, Dtype>* DataLoader::GetDenseMat()
{
	if (dense_used == (int)dense_mats.size())
	{
		dense_mats.push_back(new DenseMat<mode, Dtype>());
	}
	assert(dense_used < (int)dense_mats.size());
	dense_used++;
	return dense_mats[dense_used - 1];
}

SparseMat<mode, Dtype>* DataLoader::GetSparseMat()
{
	if (sparse_used == (int)sparse_mats.size())
	{
		sparse_mats.push_back(new SparseMat<mode, Dtype>());
	}
	assert(sparse_used < (int)sparse_mats.size());
	sparse_used++;
	return sparse_mats[sparse_used - 1];
}

std::vector< DenseMat<mode, Dtype>* > DataLoader::dense_mats;
std::vector< SparseMat<mode, Dtype>* > DataLoader::sparse_mats;
int DataLoader::dense_used;
int DataLoader::sparse_used;
