/* File: poly.i */

%include "std_vector.i"
%include "std_string.i"

namespace std {
    %template(VectorString) vector<string>;
};

%module SPOG
%{

#define SWIG_FILE_WITH_INIT
#include "spog.hh"
#include <SPOG/fv.h>
#include <vector>

%}
%include "numpy.i"
%include <std_string.i>

%init %{
    import_array();
%}

%naturalvar SPOGPoly::data;
%include "spog.hh"