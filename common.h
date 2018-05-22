#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED

#include <iostream>
#include <vector>
#include <string>

#include <dlib/clustering.h>
#include <dlib/svm_threaded.h>

static const size_t num_measurements = 7;

typedef dlib::matrix<double,num_measurements,1> sample_type;
//typedef dlib::linear_kernel<sample_type> kernel_type;
typedef dlib::radial_basis_kernel<sample_type> kernel_type;
typedef dlib::one_vs_one_trainer<dlib::any_trainer<sample_type> > type_trainer;

std::ostream& operator << (std::ostream& stream, sample_type value){
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

    if ( param == value( num_measurements-1 ) || ( 1 >= param ) ) {
        value(num_measurements-1)=0;
    } else value(num_measurements-1)=1;

    return value;
}


#endif // COMMON_H_INCLUDED
