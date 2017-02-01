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

typedef double (*ActivateFunction)(double);
typedef double (*ActivateGradFunction)(double);
typedef struct DE_Neuron{
    unsigned int size; // weight&input array size
    double * weights;  // weights
    double ** inputs;  // input pointers
    double bias;
    ActivateFunction activate;

    //for back-propagation
    double output;
    double grad;
    ActivateGradFunction GetActGrad;

    //status
    int act_type;
}DE_Neuron;

typedef struct DE_Layer{
    unsigned int input_size; // neuron array size
    unsigned int neuron_size; // neuron array size
    DE_Neuron** neurons;
}DE_Layer;

#endif //DEEPENGINE_NEURON_H
