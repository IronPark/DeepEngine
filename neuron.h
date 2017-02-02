//
// Created by LAB2 on 2017-02-01.
//

#ifndef DEEPENGINE_NEURON_H
#define DEEPENGINE_NEURON_H
#define ACT_TYPE_LINEAR 0
#define ACT_TYPE_IDENEITY 0
#define ACT_TYPE_ReLU 1
#define ACT_TYPE_ReLU 1
#define ACT_TYPE_LReLU 2
#define ACT_TYPE_SIGMOID 3
//Structs
typedef double (*ActivateFunction)(double);
typedef double (*ActivateGradFunction)(double);
typedef struct DE_Neuron{
    unsigned int size; // weight&input array size
    double * weights;  // weights
    double ** inputs;  // input pointers
    ActivateFunction activate;

    //for back-propagation
    double output;
    double grad;
    ActivateGradFunction actGrad;

    //status
    int act_type;
}DE_Neuron;

typedef struct DE_Layer{
    unsigned int neuron_size; // neuron array size
    double bias;
    DE_Neuron** neurons;
}DE_Layer;

typedef struct DE_Network{
    unsigned int NetSize; // neuron array size
    DE_Layer* layers;

}DE_Network;

//Functions

//DE_Network
DE_Network* DE_NetworkCreate();
void DE_NetworkTrain(DE_Network* neuronNet,double learningRate, double *target);

//DE_Layer
DE_Neuron* DE_LayerCreate(unsigned int input_size,unsigned int neuron_size,const int type);


#endif //DEEPENGINE_NEURON_H
