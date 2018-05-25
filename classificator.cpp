#include "common.h"
#include <map>

double distance(sample_type point1, sample_type point2){
    double dx(point2(0)-point1(0)), dy(point2(1)-point1(1));
    return dx*dx + dy*dy;
}

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
        test_file.open (filename.c_str());
        if (!test_file.good())
        {
            std::cout << "Can't open file with classificator for" << modelname << "!" << std::endl;
            return -4;
        }
        test_file.close();


        dlib::one_vs_one_decision_function<type_trainer, dlib::decision_function<kernel_type> > classificator;
        dlib::deserialize(filename) >> classificator;

        sample_type test_sample;
        std::string input;
        while(std::getline(std::cin, input))
        {
            try{
                test_sample = parse_line(input);

                long cluster = std::round(classificator(test_sample));
                //std::cout << "Cluster: " << cluster;

                filename = modelname+".c" + std::to_string(cluster);
                test_file.open(filename.c_str());
            }catch(std::exception& e){
                std::cout << "Wrong input data! " << e.what() << std::endl;
            }


            if(!test_file.good())
            {
                std::cout << "File for cluster " << filename << "  not found!";
                continue;
            }
            sample_type sample_input;
            std::multimap<double, sample_type> map_for_cluster;

            while(std::getline(test_file, input)){
                try {
                    sample_input = parse_line(input);
                }
                catch (std::exception& e)
                {
                    std::cout << "Error while reading cluster file: " << e.what() << std::endl;
                };

                double dist = distance(sample_input, test_sample);
                map_for_cluster.insert(std::make_pair(dist, sample_input));
            }
            test_file.close();

            for(auto in_cluster: map_for_cluster){
                std::cout <<in_cluster.second;
            }
        };
    }
    catch(std::exception& e){
        std::cout << "Unexpected exception! " << e.what() << std::endl;
    }
}
