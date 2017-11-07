#include "config.h"

int cfg::iter = 0;
int cfg::bptt = 0;
int cfg::warm = 0;
int cfg::num_feats = 0;
int cfg::num_entities = 0;
int cfg::num_rels = 0;
int cfg::num_samples = 1;
unsigned cfg::n_hidden = 60;
unsigned cfg::n_embed_E = 60;
unsigned cfg::n_embed_R = 60;
unsigned cfg::max_iter = 0;
unsigned cfg::test_interval = 10000;
unsigned cfg::report_interval = 100;
unsigned cfg::save_interval = 50000;
unsigned cfg::skip = 0;
Dtype cfg::time_scale = 1.0;
Dtype cfg::lr = 0.0005;
Dtype cfg::l2_reg = 0.0;
Dtype cfg::weight_scale = 0.01;
Dtype cfg::min_dur = 24; 
Dtype cfg::max_dur = 500;
const char* cfg::f_train = nullptr;
const char* cfg::f_test = nullptr;
const char* cfg::f_meta = nullptr;
const char* cfg::save_dir = "./saved";
