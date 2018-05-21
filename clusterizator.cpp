#include <iostream>
#include <vector>
#include <string>

#include <dlib/clustering.h>
#include <dlib/svm_threaded.h>

using namespace std;
using namespace dlib;

static const size_t num_measurements = 7;

typedef matrix<double,num_measurements,1> sample_type;
typedef linear_kernel<sample_type> kernel_type;

ostream& operator << (ostream& stream, sample_type value){
    std::ios_base::fmtflags fmt_fl(stream.flags());
    stream << std::fixed;
    for(size_t i=0; i<num_measurements-1; ++i){
        stream << value(i);
        stream << ";";
    }
    stream << value(num_measurements-1) << std::endl;
    stream.setf(fmt_fl);
    return stream;
}

sample_type parse_line (std::string line){
    std::stringstream sstream(line);

    double param;
    char delim;
    sample_type value;

    for(size_t i=0; i<num_measurements; ++i){
        sstream >> param;
        if(sstream.fail()){
            param=0;
            sstream.clear(std::ios_base::goodbit);
        }
        sstream >> delim;
        if(delim!=';')
            throw std::logic_error("Error in format input!");

        value(i) = param;
    };

    sstream >> param;
    /*if((0 == param) || sstream.fail() || !sstream.eof())
       throw std::logic_error("Error in format input!");*/

    if ( param == value( num_measurements-1 ) || ( 1 >= param ) ) {
        value(num_measurements-1)=0;
    } else value(num_measurements-1)=1;

    return value;
}

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

        kcentroid<kernel_type> kc(kernel_type(),0.000001, num_clusters);
        kkmeans<kernel_type> test(kc);
        std::vector<sample_type> initial_centers;

        test.set_number_of_centers(num_clusters);
        pick_initial_centers(num_clusters, initial_centers, samples, test.get_kernel());
        test.train(samples,initial_centers);

        std::vector<unsigned long> assignments = spectral_cluster(kernel_type(), samples, num_clusters);

        save_clusters_in_files(samples, assignments, modelname, num_clusters);

        typedef one_vs_one_trainer<any_trainer<sample_type> > type_trainer;
        type_trainer trainer;

        krr_trainer<kernel_type> trainer_linear;
        trainer_linear.set_kernel(kernel_type());
        trainer.set_trainer(trainer_linear);

        std::vector<double> labels;
        std::copy(assignments.begin(), assignments.end(), back_inserter(labels));
        one_vs_one_decision_function<type_trainer> des_func = trainer.train(samples, labels);
        serialize(modelname + ".dat") << des_func;

        for(size_t idx=0; idx<labels.size(); ++ idx){
            auto calc = des_func(samples.at(idx));
            auto delta = fabs(assignments.at(idx) - calc);
            std::cout << std::boolalpha << assignments.at(idx) << ";" << calc << ";" <<(delta < 0.1) << std::endl;
        }
    }
    catch(std::exception& e){
        std::cout << "Unexpected exception! " << e.what() << std::endl;
    }
}
