#include <stdio.h>
#include "network.h"
#include "parser.h"
#include "darknet.h"
#include <iostream> 
#define MAJOR_VERSION 0
#define MINOR_VERSION 2
#define PATCH_VERSION 5

using namespace std;
// #ifdef __cplusplus
// extern "C" {
// #endif

//total net type is {'CONVOLUTIONAL':save_convolutional_weights,load_convolutional_weights, 'MAXPOOL', 'ROUTE', 'SAM', 
                   //'SHORTCUT':save_shortcut_weights:load_shortcut_weights, 'UPSAMPLE', 'YOLO'}
 
char *get_name_of_layer_type(int type)
{
    switch (type)
    {
    case CONVOLUTIONAL:
        return "CONVOLUTIONAL";
    case DECONVOLUTIONAL:
        return "DECONVOLUTIONAL";
    case CONNECTED:
        return "CONNECTED";
    case MAXPOOL:
        return "MAXPOOL";
    case LOCAL_AVGPOOL:
        return "LOCAL_AVGPOOL";
    case SOFTMAX:
        return "SOFTMAX";
    case DETECTION:
        return "DETECTION";
    case DROPOUT:
        return "DROPOUT";
    case CROP:
        return "CROP";
    case ROUTE:
        return "ROUTE";
    case COST:
        return "COST";
    case NORMALIZATION:
        return "NORMALIZATION";
    case AVGPOOL:
        return "AVGPOOL";
    case LOCAL:
        return "LOCAL";
    case SHORTCUT:
        return "SHORTCUT";
    case SCALE_CHANNELS:
        return "SCALE_CHANNELS";
    case SAM:
        return "SAM";
    case ACTIVE:
        return "ACTIVE";
    case RNN:
        return "RNN";
    case GRU:
        return "GRU";
    case LSTM:
        return "LSTM";
    case CONV_LSTM:
        return "CONV_LSTM";
    case CRNN:
        return "CRNN";
    case BATCHNORM:
        return "BATCHNORM";
    case NETWORK:
        return "NETWORK";
    case XNOR:
        return "XNOR";
    case REGION:
        return "REGION";
    case YOLO:
        return "YOLO";
    case GAUSSIAN_YOLO:
        return "GAUSSIAN_YOLO";
    case ISEG:
        return "ISEG";
    case REORG:
        return "REORG";
    case REORG_OLD:
        return "REORG_OLD";
    case UPSAMPLE:
        return "UPSAMPLE";
    case LOGXENT:
        return "LOGXENT";
    case L2NORM:
        return "L2NORM";
    case EMPTY:
        return "EMPTY";
    case BLANK:
        return "BLANK";
    default:
        return "==============";
    }
}

int main(int c,char** v)
{
    char *cfg_filename = "yolov4_test.cfg";
    if(c>2){
        cfg_filename=v[1];
    }
    network net = parse_network_cfg_custom(cfg_filename, 1, 1);
    printf("the network.n is %d,\n", net.n);
    // for (int i = 0; i < net.n; i++)
    // {
    //     layer l = net.layers[i];
    //     cout << get_name_of_layer_type(l.type) << endl;
    // }

    char* weights_file="test_np_w_yolo_r.npy";
    if (c>2){
        weights_file=v[2];
        
        fprintf(stderr,"\ncfg file is assigned , and it is %s, weights file is assigned ,and it is %s\n",v[2]);
    }

    FILE* fid=fopen(weights_file,"rb");
    fprintf(stderr,"loading weights from file %s",weights_file);
    fflush(stdout);fflush(stderr);
    int major,minor,reversion;
    uint64_t iseen;
    fread(&major,sizeof(int),1,fid);
    fread(&minor,sizeof(int),1,fid);
    fread(&reversion,sizeof(int),1,fid);
    fread(&iseen,sizeof(uint64_t),1,fid);
    fprintf(stderr,"\n major is %d,minor is %d,reversion is %d, and iseen is %ld \n",major,minor,reversion,iseen);
    char* image_name="xx.jpg";
    if(c>3)image_name=v[3];
    image im = load_image("xx.jpg", 0, 0, net.c);
    float* ret=network_predict_image(&net,im);
    for(int i=0;i<10;i++)
        fprintf(stderr,"\n ret[%d] is %f%%.\n",i,ret[i]*100);
    exit(0);
}

// #ifdef __cplusplus
// }
// #endif // __cpluscplus

