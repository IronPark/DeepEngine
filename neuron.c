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
double ReLUGrad(double x){
    return x>0.0?1.0:0.0;
}
double LReLUGrad(double x){
    return x > 0.0 ? 1.0 : 0.01;
}
double LinearGrad(double x){//Ideneity
    return 1.0;
}
double SigmoidGrad(double x){//Ideneity
    return  (1.0 - x) * x;
}

DE_Neuron* DE_NeuronCreate(unsigned int size,const int type);
//DE_Network - Start
DE_Network* DE_NetworkCreate(){
    DE_Network* neuronNet = (DE_Network*) malloc(sizeof(DE_Network));
    neuronNet->layers;
}
int DE_NetworkSetLayers(DE_Layer* Format, ...){

}

//NeuronLayer - Start
DE_Neuron* DE_LayerCreate(unsigned int input_size,unsigned int neuron_size,const int type){
    DE_Layer* neuronLayer = (DE_Layer*) malloc(sizeof(DE_Layer));
    neuronLayer->neuron_size = neuron_size;
    neuronLayer->neurons =  (DE_Neuron**) malloc(sizeof(DE_Neuron*) * neuron_size);
    for (int i = 0; i < neuron_size; ++i) {
        neuronLayer->neurons[i] = DE_NeuronCreate(input_size,type);
    }
}


void DE_NetworkTrain( DE_Network* neuronNet,double learningRate, double *target ) {
    int i,j,k;
    double temp;
    DE_Layer *currLayer, *nextLayer,*outputLayer;
    unsigned int netSize = neuronNet->NetSize;
    //calculate errors-start
    //for output layer
    for ( j = 0; j < outputLayer->neuron_size; j++ ) {
        DE_Neuron* neuron = currLayer->neurons[j];
        neuron->grad = neuron->actGrad(neuron->output) * (target[j] - neuron->output);
    }

    //for other layers
    for(i = netSize - 1; i >= 0; i--) {
        currLayer = &neuronNet->layers[i];
        nextLayer = &neuronNet->layers[i+1];
        for ( j = 0; j < currLayer->neuron_size; j++ ) {
            temp = 0;
            DE_Neuron* cNeuron = currLayer->neurons[j];
            for ( k = 0; k < nextLayer->neuron_size; k++ ) {
                DE_Neuron* nLayerNeuron = nextLayer->neurons[k];
                temp += nLayerNeuron->grad * nLayerNeuron->weights[j];
            }
            cNeuron->grad = cNeuron->actGrad(cNeuron->output) * temp;
        }
    }

    //calculate errors-end
    // update weights
    double tempWeight;
    for(i = netSize - 1; i >= 0; i--) {
        currLayer = &neuronNet->layers[i];
        nextLayer = &neuronNet->layers[i+1];
        for ( j = 0; j < currLayer->neuron_size; j++ ) {
            // weights
            for( k = 0; k < currLayer->neurons[j]->size; k++ ) {
                tempWeight = currLayer->neurons[j]->weights[k];
                const double delta_w =  learningRate * currLayer->neurons[j]->grad * *(currLayer->neurons[j]->inputs[k]);
                currLayer->neurons[j]->weights[k]++;
            }
        }
    }
}

//NeuronLayer - End
DE_Neuron* DE_NeuronCreate(unsigned int size,const int type){
    DE_Neuron* neuron = (DE_Neuron*) malloc(sizeof(DE_Neuron));
    neuron->size = size;
    neuron->act_type = type;
    neuron->weights = (double*)malloc(neuron->size*sizeof(double));
    neuron->inputs  = (double**)malloc(neuron->size*sizeof(double*));
    //Set activation function for neuron
    switch (type){
        case ACT_TYPE_ReLU:
            neuron->activate = ReLU;
            neuron->actGrad  = ReLUGrad;
            break;
        case ACT_TYPE_LINEAR:
            neuron->activate = Linear;
            neuron->actGrad  = LinearGrad;
            break;
        case ACT_TYPE_SIGMOID:
            neuron->activate = Sigmoid;
            neuron->actGrad  = SigmoidGrad;
        case ACT_TYPE_LReLU:
            neuron->activate = LReLU;
            neuron->actGrad  = LReLUGrad;
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
    return neuron->output = neuron->activate(sum);
}

