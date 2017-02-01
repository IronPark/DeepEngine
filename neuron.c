//
// Created by LAB2 on 2017-02-01.
//
#include <malloc.h>
#include <math.h>
#include "neuron.h"
//Activation Functions - START
double ReLU(double x){
    return x > 0?x:0;
}
double LReLU(double x){
    return x > 0.0 ? x : 0.01*x;
}
double Linear(double x){//Ideneity
    return x;
}
double Sigmoid(double x){//Ideneity
    return 1.0 / (1.0 + exp(-x));
}
//Activation Functions - END


double GetActGrad(double x,int type);
DE_Neuron* DE_Neuron_Create(unsigned int size,const int type);

//NeuronLayer - Start
DE_Neuron* DE_Layer_Create(unsigned int input_size,unsigned int neuron_size,const int type){
    DE_Layer* neuronLayer = (DE_Layer*) malloc(sizeof(DE_Layer));
    neuronLayer->input_size = input_size;
    neuronLayer->neuron_size = neuron_size;
    neuronLayer->neurons =  (DE_Neuron**) malloc(sizeof(DE_Neuron*) * neuron_size);
    for (int i = 0; i < neuron_size; ++i) {
        neuronLayer->neurons[i] = DE_Neuron_Create(input_size,type);
    }
}

//for Output Layer
void DE_CalGradOutput(DE_Layer* Layers, double *target){
    for (int j = 0; j < Layers->neuron_size; ++j) {
        DE_Neuron* neuron = Layers->neurons[j];
        const double grad = (target[j] - neuron->output) * GetActGrad(neuron->output,neuron->act_type);
        neuron->grad = grad;
    }
}
//for Hidden Layer
void DE_CalGradHidden(DE_Layer* TargetLayers,DE_Layer* SrcLayers){
    for (int j = 0; j < TargetLayers->neuron_size; ++j) {
        DE_Neuron* neuronT = TargetLayers->neurons[j];
        double neuron_grad = 0;
        for (int j = 0; j < SrcLayers->neuron_size; ++j) {
            DE_Neuron* neuronS = SrcLayers->neurons[j];
            neuron_grad += (neuronS->weights[j] * neuronS->grad);
        }
        neuronT->grad *= GetActGrad(neuron_grad,neuronT->act_type);
    }
}

//NeuronLayer - End
DE_Neuron* DE_Neuron_Create(unsigned int size,const int type){
    DE_Neuron* neuron = (DE_Neuron*) malloc(sizeof(DE_Neuron));
    neuron->size = size;
    neuron->act_type = type;
    neuron->weights = (double*)malloc(neuron->size*sizeof(double));
    neuron->inputs  = (double**)malloc(neuron->size*sizeof(double*));
    //Set activation function for neuron
    switch (type){
        case ACT_TYPE_ReLU:
            neuron->activate = ReLU;
            break;
        case ACT_TYPE_LINEAR:
            neuron->activate = Linear;
            break;
        case ACT_TYPE_SIGMOID:
            neuron->activate = Sigmoid;
        case ACT_TYPE_LReLU:
            neuron->activate = LReLU;
            break;
        default:
            break;
    }
    return neuron;
}

double FeedForward(DE_Neuron* neuron){
    const unsigned int size = neuron->size;
    double sum;
    for (int i = 0; i < size; ++i) {
        double input = *neuron->inputs[i];
        double weight = neuron->weights[i];
        sum+= input*weight;
    }
    const double sigma = sum + neuron->bias;
    return neuron->output = neuron->activate(sigma);
}

double GetActGrad(double x,int type){
    switch (type){
        case ACT_TYPE_ReLU:
            return x>0.0?1.0:0.0;
        case ACT_TYPE_LINEAR:
            return 1.0;
        case ACT_TYPE_SIGMOID:
            return  (1.0 - x) * x;
        case ACT_TYPE_LReLU:
            return x > 0.0 ? 1.0 : 0.01;
        default:
            break;
    }
    return 0.0;
}
