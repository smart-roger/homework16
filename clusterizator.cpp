#include "common.h"

void save_clusters_in_files(std::vector<sample_type>& samples,
                            std::vector<unsigned long>& clusters,
                            std::string model_name,
                            unsigned long num_clusters){
    std::vector<std::shared_ptr<std::ofstream>> vecFiles;
    vecFiles.reserve(num_clusters);
    for(size_t cluster=0; cluster<num_clusters;++cluster){
        vecFiles.push_back(std::make_shared<std::ofstream>(model_name+".c"+std::to_string(cluster)));
    }
    for(size_t idx=0; idx < samples.size() && idx<clusters.size(); ++idx){

        unsigned long cluster(clusters.at(idx));
        sample_type sample(samples.at(idx));

        std::cout << cluster <<":"<<sample;

        std::ofstream& file(*vecFiles[cluster]);
        file << samples.at(idx);
    }
}

int main(int argc, char** argv)
{
    try{
        if(argc<3){
            std::cout << "usage rclst <num_clusters> <modelname>" << std::endl;
            return -1;
        }

        unsigned long num_clusters = std::stoul(argv[1]);
        if(num_clusters<=1){
            std::cout << "Wrong num_clusters! num_clusters should be more than 1!" << std::endl;
            return -2;
        }

        std::string modelname (argv[2]);
        if (modelname.empty()){
            std::cout << "Wrong model_name! modelname can't be empty!" << std::endl;
            return -3;
        }

        sample_type m;
        std::vector<sample_type> samples;
        std::string line;

        std::getline(std::cin, line);
        while(!std::cin.eof()){
            try{
                m = parse_line(line);
                samples.push_back(m);
            }
            catch(std::logic_error& ex){
                std::cout << "Wrong line: " << line << std::endl;
            }
            catch(std::exception& ex){
                std::cout << "Unexpected: " << ex.what() << std::endl;
            }
            std::getline(std::cin, line);
        };

        dlib::kcentroid<kernel_type> kc(kernel_type(0.0000001),0.0000001, num_clusters);
        dlib::kkmeans<kernel_type> test(kc);
        std::vector<sample_type> initial_centers;

        test.set_number_of_centers(num_clusters);
        dlib::pick_initial_centers(num_clusters, initial_centers, samples, test.get_kernel());
        test.train(samples,initial_centers);

        std::vector<unsigned long> assignments = spectral_cluster(kernel_type(0.0000001), samples, num_clusters);

        save_clusters_in_files(samples, assignments, modelname, num_clusters);

        type_trainer trainer;

        dlib::krr_trainer<kernel_type> trainer_linear;
        trainer_linear.set_kernel(kernel_type(0.0000001));
        trainer.set_trainer(trainer_linear);

        std::vector<double> labels;
        std::copy(assignments.begin(), assignments.end(), back_inserter(labels));
        dlib::one_vs_one_decision_function<type_trainer, dlib::decision_function<kernel_type> > des_func = trainer.train(samples, labels);
        dlib::serialize(modelname + ".dat") << des_func;

        /*for(size_t idx=0; idx<labels.size(); ++ idx){
            auto calc = des_func(samples.at(idx));
            auto delta = fabs(assignments.at(idx) - calc);
            if ( delta > 0.1)
                std::cout << std::boolalpha << assignments.at(idx) << ";" << calc << ";" <<(delta < 0.1) << std::endl;
        }*/
    }
    catch(std::exception& e){
        std::cout << "Unexpected exception! " << e.what() << std::endl;
    }
}
