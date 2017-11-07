#include <random>
#include <deque>
#include <queue>
#include <ctime>
#include "cppformat/format.h"
#include "config.h"
#include "dataset.h"
#include "data_loader.h"
#include "layer_holder.h"
#include "recorder.h"

#include "nngraph.h"
#include "model.h"
#include "learner.h"

#include "linear_param.h"
#include "const_scalar_param.h"
#include "param_layer.h"
#include "input_layer.h"
#include "relu_layer.h"
#include "tanh_layer.h"
#include "concat_layer.h"
#include "c_add_layer.h"
#include "c_mul_layer.h"
#include "mse_criterion_layer.h"
#include "abs_criterion_layer.h"

#include "embedding_holder.h"
#include "sparse_survival_nll_layer.h"
#include "sparse_onedim_rank_criterion_layer.h"
#include "base_intensity_layer.h"
#include "rayl_intensity_nll_layer.h"
#include "rayl_time_pred_layer.h"
#include "bipartite_graph.h"

#define max(x, y) (x > y ? x : y)


NNGraph<mode, Dtype> gnn;
Model<mode, Dtype> model;

Dataset train_data, test_data;
EmbeddingHolder  *entity_embedding;
BipartiteGraph* bg;

Recorder cur_time;
std::map<std::string, int> train_event_dict;
std::map<std::string, int> train_entity_dict;

//Added to fix prefix related bug
std::string GetLayerPrefix(std::string name)
{
	auto pos = name.find('-');
	assert(pos != std::string::npos);
	return name.substr(0, pos);
}


// Data and Config Loading Operations
inline void LoadRawData()
{
	assert(cfg::f_train && cfg::f_test && cfg::f_meta);
	cfg::LoadMetaInfo();
	train_data.LoadEvents(cfg::f_train, TRAIN);
	test_data.LoadEvents(cfg::f_test, TEST);
	std::cerr << "Events loaded" << std::endl;
	std::cerr << "******************************\n" << std::endl;


	for (size_t e_idx = 0; e_idx < test_data.ordered_events.size(); ++e_idx)
	{

		auto* cur_event = test_data.ordered_events[e_idx];
		cur_event->global_idx += train_data.ordered_events.size();
		if (train_data.entity_event_lists[cur_event->subject].size() == 0)
			cur_event->subj_prev_event = nullptr;
		else{
			assert(train_data.entity_event_lists[cur_event->subject].back()->t <= cur_event->t);
			cur_event->subj_prev_event = train_data.entity_event_lists[cur_event->subject].back();
			cur_event->subj_prev_event->subj_next_event = cur_event;
		}


		if (train_data.entity_event_lists[cur_event->object].size() == 0)
			cur_event->obj_prev_event = nullptr;
		else{
			assert(train_data.entity_event_lists[cur_event->object].back()->t <= cur_event->t);
			cur_event->obj_prev_event = train_data.entity_event_lists[cur_event->object].back();
			cur_event->obj_prev_event->obj_next_event = cur_event;
		}

	}

	// Required for filtered reporting
	for (size_t e_idx = 0; e_idx < train_data.ordered_events.size(); ++e_idx)
	{
		std::string e1 = std::to_string(train_data.ordered_events[e_idx]->subject);
		std::string e2 = std::to_string(train_data.ordered_events[e_idx]->object);
		std::string r = std::to_string(train_data.ordered_events[e_idx]->rel);
		std::string key = e1 + "_" + r + "_" + e2;
		if (train_event_dict.count(key) == 0)
		{
			train_event_dict[key]+=1;
		}

		if (e_idx >= cfg::skip)
		{
			if (train_entity_dict.count(e1) == 0)
			{
				train_entity_dict[e1]+=1;
			}

			if (train_entity_dict.count(e2) == 0)
			{
				train_entity_dict[e2]+=1;
			}
		}
	}

	std::cerr << "# train: " << train_data.ordered_events.size() << " # test: " << test_data.ordered_events.size() << std::endl;
	std::cerr << "Total number of entities: " << cfg::num_entities  << std::endl;
	std::cerr << "Total number of unique relations: " << cfg::num_rels << std::endl;
	std::cerr << "Train Map size: " << train_event_dict.size() << std::endl;
	std::cerr << "Train Entity size: " << train_entity_dict.size() << std::endl;
}

//Recurrent Layer
ILayer<mode, Dtype>* AddRecur(std::string name, 
				  ILayer<mode, Dtype>* dur_layer,
				  ILayer<mode, Dtype>* concat_layer,
				  IParam<mode, Dtype>* weight_dur,
				  IParam<mode, Dtype>* weight_input,
				  std::string suffix)
{
	return gl< ParamLayer >(name + "_rnn_" + suffix, gnn, {dur_layer, concat_layer}, {weight_dur, weight_input});
}


// Projection and Concatenation Hidden Layer to capture evolution
ILayer<mode, Dtype>* GetEvolve(std::string name, 
				   ILayer<mode, Dtype>* input_dur,
				   ILayer<mode, Dtype>* other_dynamic_embed,
				   ILayer<mode, Dtype>* prev_embed,
				   ILayer<mode, Dtype>* rel_embed,
				   std::string rel)
{
	auto& param_dict = model.diff_params;
	std::string other_name = "object";
	if (name == "object")
		other_name = "subject";

	ILayer<mode, Dtype>* concat_inp = gl< ConcatLayer >(name + "_input",gnn,{prev_embed, other_dynamic_embed, rel_embed});

	auto* hidden_concat_layer = gl< ParamLayer >("project_layer", gnn, {concat_inp}, {param_dict["w_concat_input"]});
	auto* tanh_concat_layer = gl< TanhLayer >("tanh_concat_input", gnn, {hidden_concat_layer});

	auto* hidden_candidate_linear = AddRecur(name, input_dur, tanh_concat_layer, param_dict["w_time_" + name],
						param_dict["w_input"],"to_h");

	std::string layer_prefix = GetLayerPrefix(hidden_candidate_linear->name);
	return gl< TanhLayer >("tanh_" + layer_prefix, gnn, {hidden_candidate_linear});
}

//One-hot Input
ILayer<mode, Dtype>* GetOneHotInput(std::string prefix, 
					int dim, 
					int id, 
					std::map<std::string, IMatrix<mode, Dtype>*>& inputs, 
					std::vector<ILayer<mode, Dtype>*>& lookup_table)
{
	assert(dim == (int)lookup_table.size());
	if (!lookup_table[id])
	{
		auto* one_hot_input = gl< InputLayer >(prefix + "_onehot_input", gnn, {});
		inputs[one_hot_input->name] = DataLoader::LoadOneHot(dim, id);
		lookup_table[id] = one_hot_input;
	}
	return lookup_table[id];
}


//Fetch Embedded Vector 
ILayer<mode, Dtype>* GetEmbeddingParam(std::string prefix, 
					int dim,
					int id, 
					std::map<std::string, IMatrix<mode, Dtype>*>& inputs, 
					IParam<mode, Dtype>* embeddings, 
					std::vector<ILayer<mode, Dtype>*>& onehot_lookup, 
					std::vector<ILayer<mode, Dtype>*>& embedding_lookup)
{

	assert(dim == (int)embedding_lookup.size());
	if (!embedding_lookup[id])
	{
		auto* one_hot_input = GetOneHotInput(prefix, dim, id, inputs, onehot_lookup);

		auto* linear_one_hot = gl< ParamLayer >(prefix + "_" + embeddings->name, gnn, {one_hot_input}, {embeddings});
		embedding_lookup[id] = gl< TanhLayer >(prefix + "_act_" + embeddings->name, gnn, {linear_one_hot});
	}
	return embedding_lookup[id];
}


//Update Embeddings after observing an event
std::pair<ILayer<mode, Dtype>*, ILayer<mode, Dtype>*> GetEmbedAfterEvent(Event* e,
							std::map<std::string, IMatrix<mode, Dtype>*>& inputs, 
							std::vector< ILayer<mode, Dtype>* >& lookup_entity_onehot,
							std::vector< ILayer<mode, Dtype>* >& lookup_rel_onehot,
							std::vector< ILayer<mode, Dtype>* >& lookup_entity_init,
							std::vector< ILayer<mode, Dtype>* >& lookup_rel_init)
{
	auto& param_dict = model.diff_params;

	//add for time prediction
	auto dur_feat = DataLoader::LoadDurFeat(e);

	auto* subject_dur_input  = gl< InputLayer >("subject_dur",gnn,{});
	auto* object_dur_input  = gl< InputLayer >("object_dur",gnn,{});

	inputs[subject_dur_input->name] = dur_feat.first;
	inputs[object_dur_input->name] = dur_feat.second;

	//Get previous relational embedding
	ILayer<mode, Dtype>* latest_subject_rel_embed = entity_embedding->GetLatestRelEmbedding(e->subject);
	if(!latest_subject_rel_embed)
				latest_subject_rel_embed = GetEmbeddingParam("rel", cfg::num_rels, e->rel, inputs, param_dict["w_rel_init"], lookup_rel_onehot, lookup_rel_init);

	ILayer<mode, Dtype>* latest_object_rel_embed = entity_embedding->GetLatestRelEmbedding(e->object);
	if (!latest_object_rel_embed)
			latest_object_rel_embed = GetEmbeddingParam("rel", cfg::num_rels, e->rel, inputs, param_dict["w_rel_init"], lookup_rel_onehot, lookup_rel_init);

	//Get subject/object embeddings
	ILayer<mode, Dtype>* latest_subject_embed = entity_embedding->GetLatestEmbedding(e->subject);
	if (!latest_subject_embed)
		latest_subject_embed = GetEmbeddingParam("subject", cfg::num_entities, e->subject, inputs, param_dict["w_entity_init"], lookup_entity_onehot, lookup_entity_init);

	ILayer<mode, Dtype>* latest_object_embed = entity_embedding->GetLatestEmbedding(e->object);
	if (!latest_object_embed)
		latest_object_embed = GetEmbeddingParam("object", cfg::num_entities, e->object, inputs, param_dict["w_entity_init"], lookup_entity_onehot, lookup_entity_init);

	auto* subject_new_embed = GetEvolve("subject", subject_dur_input, latest_object_embed, latest_subject_embed, latest_subject_rel_embed, std::to_string(e->rel)); 

	auto* object_new_embed = GetEvolve("object", object_dur_input, latest_subject_embed, latest_object_embed, latest_object_rel_embed, std::to_string(e->rel)); 

	ILayer<mode, Dtype>* rel_new_embed = GetEmbeddingParam("rel", cfg::num_rels, e->rel, inputs, param_dict["w_rel_init"], lookup_rel_onehot, lookup_rel_init);

	entity_embedding->InsertRelEmbedding(e->subject, e, rel_new_embed);
	entity_embedding->InsertRelEmbedding(e->object, e, rel_new_embed);

	return std::make_pair(subject_new_embed, object_new_embed);
}

// Output Layers of the Network
void BuildOutput(Recorder& cur_time, Event* cur_event, std::map<std::string, IMatrix<mode, Dtype>*>& inputs, 
				std::vector<ILayer<mode, Dtype>*>& layer_buf, bool is_last)
{
	auto& param_dict = model.diff_params;
	auto* cur_subject_embed = layer_buf[bg->entity_idx[cur_event->subject]];
	auto* cur_object_embed = layer_buf[bg->entity_idx[cur_event->object]];
	Dtype t_end = cur_event->t;
	std::string embed_name = "R_" + std::to_string(cur_event->rel);

	//Add a new layer to calculate intensity

	/*Base Intensity*/
	auto* base_intensity = gl< BaseIntensityLayer >("base_intensity", gnn, {cur_subject_embed, cur_object_embed},dynamic_cast<LinearParam<mode, Dtype>*>(param_dict[embed_name]));
	auto* dur_label = gl< InputLayer >("dur_label", gnn, {});
	auto* mat_dur_label = DataLoader::LoadLabel(cur_event->t - cur_time.GetCurTime(cur_event->subject, cur_event->object));
	inputs[dur_label->name] = mat_dur_label;

	/*Rayleigh Intensity*/
	gl< RaylIntensityNllLayer >("nll_int", gnn, {base_intensity, dur_label});

	//Changed layer for relation parameter
	auto* surv_layer = gl< SparseSurvivalNllLayer >("nll_surv", gnn, layer_buf, cfg::num_entities, cfg::num_rels, dynamic_cast<LinearParam<mode, Dtype>*>(param_dict[embed_name]));
	if (!is_last)
		dynamic_cast<SparseSurvivalNllLayer<mode, Dtype>*>(surv_layer)->SetMutableInfo(bg, cur_event->subject, cur_event->object, cur_event->rel,t_end, cur_time);
	else 
		dynamic_cast<SparseSurvivalNllLayer<mode, Dtype>*>(surv_layer)->SetMutableInfo(bg, -1, -1, -1, t_end, cur_time);


	/*Rank Layer*/
	auto* rank_layer = gl< SparseOnedimRankCriterionLayer >("avg_rank", gnn, layer_buf, cfg::num_entities, dynamic_cast<LinearParam<mode, Dtype>*>(param_dict[embed_name]), RankOrder::DESC, train_entity_dict, train_event_dict);
	dynamic_cast<SparseOnedimRankCriterionLayer<mode, Dtype>*>(rank_layer)->SetMutableInfo(bg, cur_event->subject, cur_event->object, cur_event->rel, cur_event->t, cur_time, true);

	/*Time Prediction Layers*/
	auto* time_pred_layer = gl< RaylTimePredLayer >("time_pred", gnn, {base_intensity});
	gl< ABSCriterionLayer >("mae", gnn, {time_pred_layer, dur_label}, PropErr::N);
	gl< MSECriterionLayer >("rmse", gnn, {time_pred_layer, dur_label}, PropErr::N);

}

//Build Training Network
int BuildTrainNet(Dtype T_begin, 
			std::vector<Event*>& events, std::map<std::string, IMatrix<mode, Dtype>*>& inputs,
			std::vector< ILayer<mode, Dtype>* >& lookup_entity_onehot,
			std::vector< ILayer<mode, Dtype>* >& lookup_rel_onehot,
			std::vector< ILayer<mode, Dtype>* >& lookup_entity_init,
			std::vector< ILayer<mode, Dtype>* >& lookup_rel_init)
{
	bg->Reset();
	for (size_t e_idx = 0; e_idx < events.size(); ++e_idx)
		bg->AddEvent(events[e_idx]->subject, events[e_idx]->object, true);
	
	auto& param_dict = model.diff_params;
	std::vector<ILayer<mode, Dtype>*> layer_buf;
	layer_buf.resize(bg->entity_list.size());

	for (int i = 0; i < cfg::num_entities; ++i)
	{
		lookup_entity_onehot[i] = nullptr;
		lookup_entity_init[i] = nullptr;
	}

	for (int i = 0; i < cfg::num_rels; ++i)
	{
		lookup_rel_onehot[i] = nullptr;
		lookup_rel_init[i] = nullptr;
	}

	for (size_t t = 0; t < bg->entity_list.size(); ++t)
	{
		int i = bg->entity_list[t];
		lookup_entity_init[i] = GetEmbeddingParam("entity", cfg::num_entities, i, inputs,
								param_dict["w_entity_init"], lookup_entity_onehot, lookup_entity_init);
		layer_buf[t] = lookup_entity_init[i];
	}

	cur_time.Init(cfg::num_entities, T_begin);

	for (size_t e_idx = 0; e_idx < events.size(); ++e_idx)
	{		
		auto* cur_event = events[e_idx];
		BuildOutput(cur_time, cur_event, inputs, layer_buf, e_idx + 1 == events.size());

		if (e_idx + 1 == events.size())
			break;

		auto p = GetEmbedAfterEvent(cur_event, inputs, 
									lookup_entity_onehot, lookup_rel_onehot,
									lookup_entity_init, lookup_rel_init);

		entity_embedding->InsertEmbedding(cur_event->subject, cur_event, p.first);
		entity_embedding->InsertEmbedding(cur_event->object, cur_event, p.second);

		layer_buf[bg->entity_idx[cur_event->subject]] = p.first;
		layer_buf[bg->entity_idx[cur_event->object]] = p.second;
		cur_time.UpdateEvent(cur_event->subject, cur_event->object, cur_event->t);
	}

	return events.size();
}

//Fetch Mini-Batch Sequence to train
void GetMiniBatch_SEQ(Event* e_end, std::vector<Event*>& event_mini_batch)
{
	event_mini_batch.clear();
	for (int i = max(e_end->global_idx - cfg::bptt + 1, 0); i <= e_end->global_idx; ++i)
	{
		event_mini_batch.push_back(train_data.ordered_events[i]);
	}
}


//Setup Test, required for initializations
void StartupTest(std::vector<ILayer<mode, Dtype>*>& latest_embeddings, 
				std::vector< ILayer<mode, Dtype>* >& lookup_entity_onehot,
				std::vector< ILayer<mode, Dtype>* >& lookup_rel_onehot,
				std::vector< ILayer<mode, Dtype>* >& lookup_entity_init,
				std::vector< ILayer<mode, Dtype>* >& lookup_rel_init)
{
	std::map<std::string, IMatrix<mode, Dtype>*> inputs;
	bg->Reset();

	auto& param_dict = model.diff_params;

	for (size_t e_idx = 0; e_idx < train_data.ordered_events.size(); ++e_idx)
		bg->AddEvent(train_data.ordered_events[e_idx]->subject, train_data.ordered_events[e_idx]->object, true);
	for (size_t e_idx = 0; e_idx < test_data.ordered_events.size(); ++e_idx)
		bg->AddEvent(test_data.ordered_events[e_idx]->subject, test_data.ordered_events[e_idx]->object, true);

	latest_embeddings.resize(bg->entity_list.size());

	std::cerr << "Entity List Size:" << bg->entity_list.size() << std::endl;

	for (int i = 0; i < cfg::num_entities; ++i)
	{
		lookup_entity_onehot[i] = nullptr;
		lookup_entity_init[i] = nullptr;
	}

	for (int i = 0; i < cfg::num_rels; ++i)
	{
		lookup_rel_onehot[i] = nullptr;
		lookup_rel_init[i] = nullptr;
	}

	cur_time.Init(cfg::num_entities, 0);

	for (size_t e_idx = 0; e_idx < train_data.ordered_events.size(); ++e_idx)
	{
		inputs.clear();


		auto* cur_event = train_data.ordered_events[e_idx];
		auto p = GetEmbedAfterEvent(cur_event, inputs, 
						lookup_entity_onehot, lookup_rel_onehot,
						lookup_entity_init, lookup_rel_init);
		entity_embedding->InsertEmbedding(cur_event->subject, cur_event, p.first);
		entity_embedding->InsertEmbedding(cur_event->object, cur_event, p.second);
		
		latest_embeddings[bg->entity_idx[cur_event->subject]] = p.first;
		latest_embeddings[bg->entity_idx[cur_event->object]] = p.second;
		cur_time.UpdateEvent(cur_event->subject, cur_event->object, cur_event->t);

		lookup_entity_init[cur_event->subject] = nullptr;
		lookup_entity_onehot[cur_event->subject] = nullptr;
		lookup_entity_init[cur_event->object] = nullptr;
		lookup_entity_onehot[cur_event->object] = nullptr;
		lookup_rel_init[cur_event->rel] = nullptr;
		lookup_rel_onehot[cur_event->rel] = nullptr;

		gnn.FeedForward(inputs, TEST);

		for (auto it = gnn.layer_dict.begin(); it != gnn.layer_dict.end(); ++it)
		{
			if (it->first == p.first->name || it->first == p.second->name)
				continue;
			LayerHolder::RecycleLayer(it->second);
		}
		gnn.Clear();
		DataLoader::Reset();

	}
}


//Testing
void TestLoop(std::vector<ILayer<mode, Dtype>*>& latest_embeddings,
			std::vector< ILayer<mode, Dtype>* >& lookup_entity_onehot,
			std::vector< ILayer<mode, Dtype>* >& lookup_rel_onehot,
			std::vector< ILayer<mode, Dtype>* >& lookup_entity_init,
			std::vector< ILayer<mode, Dtype>* >& lookup_rel_init)
{
	std::map<std::string, IMatrix<mode, Dtype>*> inputs;
	//FILE* fid = fopen("rank_file.txt","w"); //Creates a file with predictions for each test.
	std::vector<float> sdevs;

	for (int i = 0; i < cfg::num_entities; ++i)
	{
		lookup_entity_onehot[i] = nullptr;
		lookup_entity_init[i] = nullptr;
	}

	for (int i = 0; i < cfg::num_rels; ++i)
	{
		lookup_rel_onehot[i] = nullptr;
		lookup_rel_init[i] = nullptr;
	}

	Dtype nll = 0.0, avg_rank = 0.0, mae = 0.0, hits10 = 0.0, dev=0.0, sdev=0.0, var = 0.0, sd=0.0;
	int tested = 0;
	std::vector<Dtype> buckets{0.4705,0.5089,0.5449,0.5833,0.6193,0.6553,0.6913,0.7297,0.7657,0.8017,0.8377,0.8760};
	int timept = 0;
	int cnt=0;
	std::cerr << fmt::sprintf("\nTotal events: %d", test_data.ordered_events.size()) << std::endl;
	clock_t startTime = clock();
	for (size_t e_idx = 0; e_idx < test_data.ordered_events.size(); ++e_idx)
	{

		auto* cur_event = test_data.ordered_events[e_idx];
		inputs.clear();
		tested++;
		cnt++;

		if (cur_event->t > buckets[timept])
		{
			clock_t test_time = double( clock() - startTime ) / (double)CLOCKS_PER_SEC;
			avg_rank /= (cnt);
			mae /= (cnt);
			hits10 /= (cnt);

			for(int s = 0; s <= cnt; ++s){
					dev = (sdevs[s] - avg_rank)*(sdevs[s] - avg_rank);
					sdev = sdev + dev;
			}

			var = sdev / (cnt); 
			sd = sqrt(var);

			std::cerr << fmt::sprintf("\nWeek: %d", timept) << std::endl;
			std::cerr << fmt::sprintf("\nEvents: %d", cnt) << std::endl;
			std::cerr << fmt::sprintf("\nTest Time: %d", test_time) << "secs." << std::endl;
			std::cerr << fmt::sprintf("\navg_rank: %.4f\tError: %.4f\thits@10: %4f\tmae: %.6f", avg_rank, sd, hits10, mae) << std::endl;
			avg_rank = 0.0;
			mae = 0.0;
			hits10 = 0.0;
			cnt = 0;
			dev = 0.0;
			sdev = 0.0;
			var = 0.0;
			sd = 0.0;
			sdevs.clear();
			timept = timept + 1;
			startTime = clock();
		}
		BuildOutput(cur_time, cur_event, inputs, latest_embeddings, e_idx + 1 == test_data.ordered_events.size());

		auto p = GetEmbedAfterEvent(cur_event, inputs,
									lookup_entity_onehot, lookup_rel_onehot,
									lookup_entity_init, lookup_rel_init);
		

		entity_embedding->InsertEmbedding(cur_event->subject, cur_event, p.first);
                entity_embedding->InsertEmbedding(cur_event->object, cur_event, p.second);

               	latest_embeddings[bg->entity_idx[cur_event->subject]] = p.first;
                latest_embeddings[bg->entity_idx[cur_event->object]] = p.second;

		cur_time.UpdateEvent(cur_event->subject, cur_event->object, cur_event->t);

		gnn.FeedForward(inputs, TEST);
		auto loss_map = gnn.GetLoss();
		for (auto it = loss_map.begin(); it != loss_map.end(); ++it)
		{
			if (it->first[0] == 'n')
			{
				nll += it->second;
			}
			if (it->first[0] == 'a'){
				avg_rank += it->second;
				//fprintf(fid,"%.4f\t",it->second); //Uncomment to produce rank file for entities.
				//fprintf(fid,"\n");
				if(it->second <= 10.0)
					hits10 += 1;

				sdevs.push_back(it->second);
			}
			if (it->first[0] == 'm')
				mae += it->second;
		}

		for (auto it = gnn.layer_dict.begin(); it != gnn.layer_dict.end(); ++it)
		{
			if (it->first == p.first->name || it->first == p.second->name)
				continue;
			LayerHolder::RecycleLayer(it->second);
		}

		gnn.Clear();
		DataLoader::Reset();

		if (tested % 1 == -1)
		{
			printf("%.2f%% test nll: %.4f\tavg_rank: %.4f\thits@10: %4f\tmae: %.4f\n",
				100 * (float)tested / test_data.ordered_events.size(),
				nll, avg_rank / (tested-1), hits10/(tested-1), mae / (tested-1));
		}
	}

	avg_rank /= (cnt);
	mae /= (cnt);
	hits10 /= (cnt);

	for(int s = 0; s <= cnt-1; ++s){
			dev = (sdevs[s] - avg_rank)*(sdevs[s] - avg_rank);
			sdev = sdev + dev;
	}

	var = sdev / (cnt);
	sd = sqrt(var);

	std::cerr << fmt::sprintf("\nWeek: %d", timept) << std::endl;
	std::cerr << fmt::sprintf("\navg_rank: %.4f\tError: %.4f\thits@10: %4f\tmae: %.6f", avg_rank, sd, hits10, mae) << std::endl;
	avg_rank = 0.0;
	mae = 0.0;
	hits10 = 0.0;
	cnt = 0;
	dev = 0.0;
	sdev = 0.0;
	var = 0.0;
	sd = 0.0;
	sdevs.clear();
	timept = timept + 1;
}

//Main function to run training and testing modules
void MainLoop()
{

	clock_t t;
	AdamLearner<mode, Dtype> learner(&model, cfg::lr, cfg::l2_reg);

    	std::map<std::string, IMatrix<mode, Dtype>*> inputs;

	int max_iter = (long long)cfg::max_iter;
	int init_iter = cfg::iter;
	if (cfg::warm == 1)
	{
		std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
		model.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
	}

	std::vector< ILayer<mode, Dtype>* > latest_embeddings, lookup_entity_onehot, lookup_entity_init, lookup_rel_onehot, lookup_rel_init;
	std::vector<Event*> event_mini_batch;

	lookup_entity_onehot.resize(cfg::num_entities);
	lookup_entity_init.resize(cfg::num_entities);


	lookup_rel_onehot.resize(cfg::num_rels);
	lookup_rel_init.resize(cfg::num_rels);
	int cur_pos = cfg::skip;
	t = clock();

	std::cerr << "\nTraining Start" << std::endl;
	std::cerr << "=================\n" << std::endl;
	for (; cfg::iter <= max_iter; ++cfg::iter)
	{
		if (cfg::iter!=0 and cfg::iter % cfg::test_interval == 0)
		{	

	    	t = clock()-t;
			std::cerr << "\nSetup testing" << std::endl;
			std::cerr << "=================\n" << std::endl;
			StartupTest(latest_embeddings,
						lookup_entity_onehot, lookup_rel_onehot,
						lookup_entity_init, lookup_rel_init);
			std::cerr << "\nTesting Start" << std::endl;
			std::cerr << "=================\n" << std::endl;
			TestLoop(latest_embeddings,
					lookup_entity_onehot, lookup_rel_onehot,
					lookup_entity_init, lookup_rel_init);
			std::cerr << "done" << std::endl;

			entity_embedding->ClearFull();
			t=clock();
		}

		Dtype T_begin = 0;
		if (cur_pos)
			T_begin = train_data.ordered_events[cur_pos - 1]->t;


		Event* e;
		if (cur_pos + cfg::bptt >= (int)train_data.ordered_events.size())
		{
			e = train_data.ordered_events.back();
			cur_pos = cfg::skip;
		} else
		{
			e = train_data.ordered_events[cur_pos + cfg::bptt - 1];
			cur_pos += cfg::bptt;
		}

		inputs.clear();
		GetMiniBatch_SEQ(e, event_mini_batch);

		int train_in_batch = BuildTrainNet(T_begin, event_mini_batch, inputs, 
							lookup_entity_onehot, lookup_rel_onehot,
							lookup_entity_init, lookup_rel_init);


		gnn.FeedForward(inputs, TRAIN);
		auto loss_map = gnn.GetLoss();


    		if (cfg::iter % cfg::report_interval == 0)
		{		
			Dtype nll = 0.0, avg_rank = 0.0, mae = 0.0, rmse = 0.0;
			for (auto it = loss_map.begin(); it != loss_map.end(); ++it)
			{
				if (it->first[0] == 'r')
					rmse += it->second;
				if (it->first[0] == 'n')
					nll += it->second;
				if (it->first[0] == 'a')
					avg_rank += it->second;
				if (it->first[0] == 'm')
					mae += it->second;
			}
			avg_rank /= train_in_batch;
			mae /= train_in_batch;
			rmse = sqrt(rmse / train_in_batch);
			std::cerr << fmt::sprintf("iter: %d\tnll: %.4f\tavg_rank: %.4f\tmae: %.4f\trmse: %.4f", cfg::iter, nll, avg_rank, mae, rmse) << std::endl;
		}
		gnn.BackPropagation();
		learner.Update();

		if (cfg::iter !=0 and cfg::iter % cfg::max_iter == 0)
		{
			printf("saving model for iter=%d\n", cfg::iter);
			model.Save(fmt::sprintf("%s/kevolve.model-1", cfg::save_dir, cfg::iter));
		}

		for (auto it = gnn.layer_dict.begin(); it != gnn.layer_dict.end(); ++it)
			LayerHolder::RecycleLayer(it->second);

		gnn.Clear();
		DataLoader::Reset();
		entity_embedding->ClearAll();
	}	
}

// Load Paramertrs 
void LoadParam(IParam<mode, Dtype>* param, std::ifstream& fid)
{
	return;
	auto* p = dynamic_cast<LinearParam<mode, Dtype>*>(param);
	auto& mat = p->p["weight"]->value;
	for (size_t i = 0; i < mat.count; ++i)
	{
		fid >> mat.data[i];
	}
}

void InitParams()
{
	// init embed
	//Entity embedding
	add_diff< LinearParam >(model, "w_entity_init", cfg::num_entities, cfg::n_embed_E, 0, cfg::weight_scale, BiasOption::NONE,true);

	//Relation embedding
	add_diff< LinearParam >(model, "w_rel_init", cfg::num_rels, cfg::n_embed_R, 0, cfg::weight_scale, BiasOption::NONE,false);
	
	//input time feature
	add_diff< LinearParam >(model, "w_time_subject", 1, cfg::n_hidden, 0, cfg::weight_scale, BiasOption::NONE,false);
	add_diff< LinearParam >(model, "w_time_object", 1, cfg::n_hidden, 0, cfg::weight_scale, BiasOption::NONE,false);

	// input feature to subject and object enitities
	add_diff< LinearParam >(model, "w_concat_input", 2*cfg::n_embed_E + cfg::n_embed_R, cfg::n_hidden, 0, cfg::weight_scale, BiasOption::NONE,false);
	add_diff< LinearParam >(model, "w_input", cfg::n_hidden, cfg::n_hidden, 0, cfg::weight_scale, BiasOption::NONE,false);
	
	// input feature to rel
	add_diff< LinearParam >(model, "w_input_rel", cfg::n_embed_R, cfg::n_hidden, 0, cfg::weight_scale, BiasOption::NONE,false);

	//Relation Weights
	for (size_t i = 0; i < (unsigned)cfg::num_rels; ++i)
	{
		std::string embed_name = "R_" + std::to_string(i);
		add_diff< LinearParam >(model, embed_name, cfg::n_embed_E, cfg::n_embed_E, 0, cfg::weight_scale, BiasOption::NONE,false);
	}

}

//Start Here
int main(const int argc, const char** argv)
{

	cfg::LoadParams(argc, argv);
	LoadRawData();	
	bg = new BipartiteGraph(cfg::num_entities);

	std::cerr << "Data loaded" << std::endl;
	InitParams();
	std::cerr << "Param created" << std::endl;
	DataLoader::Init();
	entity_embedding = new EmbeddingHolder(cfg::num_entities);

	LayerHolder::Init();

	MainLoop();

	return 0;	
}
