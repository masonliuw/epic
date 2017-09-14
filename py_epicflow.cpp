#include "Python.h"
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include <iostream>

#include "cpp/epic.h"
#include "cpp/image.h"
#include "cpp/io.h"
#include "cpp/variational.h"

using namespace boost::python;
using std::cout;
using std::string;

static PyObject* run_epicflow(string img1p, string img2p, string edgesp,
                              string matchesp, string outp) {
    // read arguments
    color_image_t *im1 = color_image_load(img1p.c_str());
    color_image_t *im2 = color_image_load(img2p.c_str());
    float_image edges = read_edges(edgesp.c_str(), im1->width, im1->height);
    float_image matches = read_matches(matchesp.c_str());
    const char *outputfile = outp.c_str();

    // prepare variables
    epic_params_t epic_params;
    epic_params_default(&epic_params);
    variational_params_t flow_params;
    variational_params_default(&flow_params);
    image_t *wx = image_new(im1->width, im1->height), *wy = image_new(im1->width, im1->height);
    
    // set params (KITTI optimized)
    strcpy(epic_params.method, "NW");
    epic_params.pref_nn= 25;
    epic_params.nn= 160;
    epic_params.coef_kernel = 1.1f;
    flow_params.niter_outer = 2;
    flow_params.alpha = 1.0f;
    flow_params.gamma = 0.77f;
    flow_params.delta = 0.0f;
    flow_params.sigma = 1.7f;

    // compute interpolation and energy minimization
    color_image_t *imlab = rgb_to_lab(im1);
    epic(wx, wy, imlab, &matches, &edges, &epic_params, 1);
    // energy minimization
    variational(wx, wy, im1, im2, &flow_params);
    // write output file and free memory
    writeFlowFile(outputfile, wx, wy);

    npy_intp dims[3]{wx->height, wx->width, 2};
    PyObject* final_flow = PyArray_SimpleNew(3, dims, NPY_FLOAT);
    const int h = wx->height, w = wx->width;
    float* flow_data = new float[2 * h * w];
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
          flow_data[2*y*w + 2*x] = wx->data[y*wx->stride + x];
          flow_data[2*y*w + 2*x + 1] = wy->data[y*wy->stride + x]; 
      }
    }
    memcpy(PyArray_DATA(final_flow), flow_data, sizeof(float) * 2 * h * w);
    delete [] flow_data;
    color_image_delete(im1);
    color_image_delete(imlab);
    color_image_delete(im2);
    free(matches.pixels);
    free(edges.pixels);
    image_delete(wx);
    image_delete(wy);
    return final_flow;
}

BOOST_PYTHON_MODULE(pyEpicFlow) {
  import_array();
  def("run_epicflow", &run_epicflow);
}
