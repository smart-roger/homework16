#include "common.h"
/*
void save_clusters_in_files(std::vector<sample_type>& samples,
                            std::vector<unsigned long>& clusters,
                            std::string model_name,
                            unsigned long num_clusters){
    std::vector<std::shared_ptr<ofstream>> vecFiles;
    vecFiles.reserve(num_clusters);
    for(size_t cluster=0; cluster<num_clusters;++cluster){
        vecFiles.push_back(std::make_shared<ofstream>(model_name+".c"+std::to_string(cluster)));
    }
    for(size_t idx=0; idx < samples.size() && idx<clusters.size(); ++idx){

        unsigned long cluster(clusters.at(idx));
        sample_type sample(samples.at(idx));

        std::cout << cluster <<":"<<sample;

        std::ofstream& file(*vecFiles[cluster]);
        file << samples.at(idx);
    }
}
*/

int main(int argc, char** argv)
{
    try{
        if(argc<2){
            std::cout << "usage rclst <modelname>" << std::endl;
            return -1;
        }

        std::string modelname (argv[1]);
        if (modelname.empty()){
            std::cout << "Wrong model_name! modelname can't be empty!" << std::endl;
            return -3;
        }

        std::string filename (modelname+".dat");
        std::ifstream test_file;
        test_file.open (filename);
        if (!test_file.good())
        {
            std::cout << "Can't open file with classificator for" << modelname << "!" << std::endl;
            return -4;
        }
        test_file.close();


        one_vs_one_decision_function<type_trainer, decision_function<kernel_type> > classificator;
        deserialize(filename) >> classificator;

        sample_type test_sample;
        std::cin >> test_sample;

        auto cluster = classificator(test_sample);
        std::cout << "Cluster: " << cluster;
    }
    catch(std::exception& e){
        std::cout << "Unexpected exception! " << e.what() << std::endl;
    }
}
