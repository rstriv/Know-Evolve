#ifndef cfg_H
#define cfg_H

typedef double Dtype;
#include "imatrix.h"

const MatMode mode = CPU;
#include <iostream>
#include "cppformat/format.h"
#include <cstring>
#include <fstream>
#include <set>
#include <map>

struct cfg
{

    static int iter, bptt;
    static int warm;
    static int num_feats, num_entities, num_rels, num_samples; //num_subjects, num_objects;
    static unsigned n_embed_E;
    static unsigned n_embed_R;
    static unsigned n_hidden;
    static unsigned max_iter;
    static unsigned test_interval;
    static unsigned report_interval;
    static unsigned save_interval;
    static unsigned skip;
    static Dtype lr;
    static Dtype min_dur, max_dur;
    static Dtype l2_reg;
    static Dtype weight_scale;
    static Dtype time_scale;
    static const char *f_test, *f_train, *f_meta, *f_static_feat, *save_dir;

    static void LoadMetaInfo()
    {
        std::ifstream fs(f_meta);
        fs >> num_entities >> num_rels >> num_feats;
    }

    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-bptt") == 0)
                bptt = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-meta") == 0)
                f_meta = argv[i + 1];
            if (strcmp(argv[i], "-train") == 0)
                f_train = argv[i + 1];
		    if (strcmp(argv[i], "-test") == 0)
		        f_test = argv[i + 1];
            if (strcmp(argv[i], "-t_scale") == 0)
                time_scale = atof(argv[i + 1]);
		    if (strcmp(argv[i], "-lr") == 0)
		        lr = atof(argv[i + 1]);
            if (strcmp(argv[i], "-cur_iter") == 0)
                iter = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-embed_E") == 0)
			    n_embed_E = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-embed_R") == 0)
			    n_embed_R = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-hidden") == 0)
		    	n_hidden = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-max_iter") == 0)
	       		max_iter = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-int_test") == 0)
    			test_interval = atoi(argv[i + 1]);
    	   	if (strcmp(argv[i], "-int_report") == 0)
    			report_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-int_save") == 0)
    			save_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-l2") == 0)
    			l2_reg = atof(argv[i + 1]);
            if (strcmp(argv[i], "-w_scale") == 0)
                weight_scale = atof(argv[i + 1]);
    		if (strcmp(argv[i], "-svdir") == 0)
    			save_dir = argv[i + 1];
    		if (strcmp(argv[i], "-warm") == 0)
    			warm = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-n_samples") == 0)
    			warm = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-min_dur") == 0)
    			warm = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-max_dur") == 0)
    			warm = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-skip") == 0)
    			skip = atoi(argv[i + 1]);
        }

        std::cerr<<"************************"<<std::endl;
        std::cerr<<"Parameter Initialization"<<std::endl;
        std::cerr<<"************************"<<std::endl;

        std::cerr << "cur_iter = " << iter << std::endl;
        std::cerr << "max_iter = " << max_iter << std::endl;
        std::cerr << "bptt = " << bptt << std::endl;
        std::cerr << "learning rate = " << lr << std::endl;
        std::cerr << "l2_penalty = " << l2_reg << std::endl;
        std::cerr << "negative samples = " << num_samples << std::endl;
        std::cerr << "skip events = " << skip << std::endl;

        std::cerr << "n_embed_E = " << n_embed_E << std::endl;
        std::cerr << "n_embed_R = " << n_embed_R << std::endl;
        std::cerr << "n_hidden = " << n_hidden << std::endl;
        std::cerr << "warm start = " << warm << std::endl;

        std::cerr << "min. duration = " << min_dur << std::endl;
        std::cerr << "max. duration = " << max_dur << std::endl;
        std::cerr << "time scale = " << time_scale << std::endl;
        std::cerr << "weight_scale = " << weight_scale << std::endl;

    	std::cerr << "test_interval = " << test_interval << std::endl;
    	std::cerr << "report_interval = " << report_interval << std::endl;
    	std::cerr << "save_interval = " << save_interval << std::endl;

    	std::cerr << "meta file = " << f_meta << std::endl;
    	std::cerr << "train file = " << f_train << std::endl;
    	std::cerr << "test file = " << f_test << std::endl;
    	std::cerr << "Model folder = " << save_dir << std::endl;
    	std::cerr << "*******************************" << std::endl;
    }
};

#endif
