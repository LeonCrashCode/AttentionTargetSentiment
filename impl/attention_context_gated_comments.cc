#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/program_options.hpp>

#include <unordered_map>
#include <unordered_set>

using namespace std; //using cpp standard library
using namespace dynet; //using dynet library
namespace po = boost::program_options; //using propgram options port of boost library

//float pdrop = 0.02;
float pdrop = 0.5; //the probability of dropout
float unk_prob = 0.1; // the probability of replace word in training singleton with UNK, aiming to train UNK
bool DEBUG = 0; // DEBUG or not

unsigned WORD_DIM = 200; // word embedding dimension 
unsigned HIDDEN_DIM = 150; // LSTM hidden dimension
unsigned TAG_HIDDEN_DIM = 64; // tag non-linear hidden dimension
unsigned LAYERS = 1; // LSTM layer
unsigned VOCAB_SIZE = 0; // vocabulary size
float THRESHOLD = 0.5; // the threshold to distinguish positive sentiment and negatives
unsigned ATTENTION_HIDDEN_DIM = 100; // attention hidden dimension
float noscore = 10000; // not used
dynet::Dict wd; // word dictionary to store words

int kUNK; 
unordered_map<unsigned, vector<float> > pretrained; //storing pretrained word embeddings, (first, second): first is word index, second is embeddings
vector<float> unk_embedding; // unk embeddings

unordered_map<unsigned, float> lex_dict; //not used
unsigned train_neu; // the number of neutron sentence in training data
unsigned train_pos; // the number of positive sentence in training data
unsigned train_neg; // the number of negative sentence in training data

unsigned dev_neu; // the number of neutron sentence in dev data
unsigned dev_pos; // the number of positive sentence in dev data
unsigned dev_neg; // the number of negative sentence in dev data

unsigned tst_neu; // the number of neutron sentence in test data
unsigned tst_pos; // the number of positive sentence in test data
unsigned tst_neg; // the number of negative sentence in test data
/*
input:
    argc: the numner of parameter
    argv: the parameters
    conf: the variable storing parameters
output:
    none
*/
void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data", po::value<string>(), "Test corpus")
        ("pdrop", po::value<float>()->default_value(0.5), "dropout probabilty")
	("unk_prob,u", po::value<float>()->default_value(0.1), "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("word_dim", po::value<unsigned>()->default_value(200), "word embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(150), "hidden dimension")
        ("tag_hidden_dim", po::value<unsigned>()->default_value(64), "tag hidden dimension")
	("layers", po::value<unsigned>()->default_value(1), "layers")
	("test,t", "Should training be run?")
        ("pretrained,w", po::value<string>(), "Pretrained word embeddings")
        ("lexicon", po::value<string>(), "Sentiment Lexicon")
	("train_methods", po::value<unsigned>()->default_value(0), "0 for simple, 1 for mon, 2 for adagrad, 3 for adam")
	("report_i", po::value<unsigned>()->default_value(100), "report i")
        ("dev_report_i", po::value<unsigned>()->default_value(10), "dev report i")
	("count_limit", po::value<unsigned>()->default_value(50), "count limit")
	("threshold", po::value<float>()->default_value(0.5),"[-threshold, threshold] is neutron")
	("debug", "Debug to output trace")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0 || conf->count("dev_data") == 0 || conf->count("test_data") == 0) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
}

/*
make string lowercase
input:
    line: input string
output:
    none
*/
void normalize_digital_lower(
    string& line // input words
    ){
  for(unsigned i = 0; i < line.size(); i ++){
    if(line[i] >= 'A' && line[i] <= 'Z'){
      line[i] = line[i] - 'A' + 'a';
    }
  }
}

/*
the Instance class for training, development and test dataset
*/
class Instance{
public:
	vector<unsigned> raws; // the index of raw words
	vector<unsigned> lows; // the index of lowercase of words

	vector<unsigned> words; // the index of words, which could be replace with UNK during training
	vector<float> lex_score; // not used
	int label;
	unsigned ts,te; // the start and end index (included) of target phrases 
	
	Instance(){clear();};
        ~Instance(){};
	void clear(){
		raws.clear();
		lows.clear();
		words.clear();
		lex_score.clear();
	}
    /*
    sequencitally output instance
    */
	friend ostream& operator << (
        ostream& out, // output stream
        Instance& instance // the instance being writting
        ){
		for(unsigned i = 0; i < instance.raws.size(); i ++){
			out << wd.convert(instance.raws[i]) << "/"
			    << wd.convert(instance.lows[i]) << "/";
			if(instance.lex_score[i] == noscore) out<<"n ";
			else out<<instance.lex_score[i]<<" ";
		}
		out<<" ||| ";
		out<<instance.ts<<" "<<instance.te;
		out<<" ||| ";
		out<<instance.label;
		out<<"\n";
		return out;
	}
    /*
    from the string to initialized instance
    input:
        line: input string
        neu: neutron label counting 
        pos: positive label counting
        neg: negative lable counting
    output:
        none
    */
	void load(const string& line, // one sentence
        unsigned& neu, // the number of neutron sentence
        unsigned& pos, // the number of positive sentence
        unsigned& neg // the number of negative sentence
        ){
                istringstream in(line);
                string word;
                while(in>>word) {
                        if(word == "|||") break;
                        raws.push_back(wd.convert(word));
                        normalize_digital_lower(word);
                        lows.push_back(wd.convert(word));
                }
                in>>ts;
                in>>te;
                in>>word;
                assert(word == "|||");
                in>>label;
                words = raws;
                lex_score.resize(words.size());
                if(label == 0) ++neu;
                else if(label == 1) ++pos;
                else if(label == -1) ++neg;
        }
	unsigned size(){assert(raws.size() == lows.size()); return raws.size();}
};

struct LSTMClassifier {
    LookupParameter p_word; // word lookup table 

    

    Parameter p_lbias; // not used
    Parameter p_tag2label; //not used

    Parameter p_start; // the padding for start of sentence
    Parameter p_end; // the padding for end of sentence

    LSTMBuilder l2rbuilder; // forward LSTM
    LSTMBuilder r2lbuilder; // backward LSTM

    // parameters of the whole sentence attention
    Parameter p_attbias; // attention bias
    Parameter p_input2att; // matrix from input representation to attention hidden
    Parameter p_target2att; // matrix from target representation to attention hidden
    Parameter p_att2attexp; // attention sum
   
    // paramenters of leftside phrase of target attention
    Parameter p_attbias_l; 
    Parameter p_input2att_l;
    Parameter p_target2att_l;
    Parameter p_att2attexp_l;

    // parameters of the rightside phrase of target attention
    Parameter p_attbias_r;
    Parameter p_input2att_r;
    Parameter p_target2att_r;
    Parameter p_att2attexp_r;
    
    // leftside attention hidden layer with target to non-linear hidden layer
    Parameter p_lh2zl;
    Parameter p_th2zl;
    Parameter p_zlbias;

    // rightside attention hidden layer with target to non-linear hidden layer
    Parameter p_rh2zr;
    Parameter p_th2zr;
    Parameter p_zrbias;

    // the whole sentence attention hidden layer with target to non-linear hidden layer
    Parameter p_lrh2zlr;
    Parameter p_th2zlr;
    Parameter p_zlrbias;

    // three-way gate representation to label layer
    Parameter p_final_lrR;
    Parameter p_bias;

    float zero = 0;
    float one = 1.0;

    /*
    construct function
    input:
        model: neural network model
    */
    explicit LSTMClassifier(Model& model) :
        l2rbuilder(LAYERS, WORD_DIM , HIDDEN_DIM, &model),
        r2lbuilder(LAYERS, WORD_DIM , HIDDEN_DIM, &model)
    {
        // parameters instance and initialization
        p_word   = model.add_lookup_parameters(VOCAB_SIZE, {WORD_DIM});

	p_bias = model.add_parameters({3});

	p_tag2label = model.add_parameters({1, TAG_HIDDEN_DIM});
	p_lbias = model.add_parameters({1});

        p_start = model.add_parameters({WORD_DIM});
	p_end = model.add_parameters({WORD_DIM});

	p_attbias = model.add_parameters({ATTENTION_HIDDEN_DIM});
	p_input2att = model.add_parameters({ATTENTION_HIDDEN_DIM, 2*HIDDEN_DIM});
	p_target2att = model.add_parameters({ATTENTION_HIDDEN_DIM, 2*HIDDEN_DIM});
	p_att2attexp = model.add_parameters({ATTENTION_HIDDEN_DIM});
     
	p_attbias_l = model.add_parameters({ATTENTION_HIDDEN_DIM});
        p_input2att_l = model.add_parameters({ATTENTION_HIDDEN_DIM, 2*HIDDEN_DIM});
        p_target2att_l = model.add_parameters({ATTENTION_HIDDEN_DIM, 2*HIDDEN_DIM});
        p_att2attexp_l = model.add_parameters({ATTENTION_HIDDEN_DIM});

	p_attbias_r = model.add_parameters({ATTENTION_HIDDEN_DIM});
        p_input2att_r = model.add_parameters({ATTENTION_HIDDEN_DIM, 2*HIDDEN_DIM});
        p_target2att_r = model.add_parameters({ATTENTION_HIDDEN_DIM, 2*HIDDEN_DIM});
        p_att2attexp_r = model.add_parameters({ATTENTION_HIDDEN_DIM});

	p_lh2zl = model.add_parameters({2*HIDDEN_DIM, 2*HIDDEN_DIM});
        p_th2zl = model.add_parameters({2*HIDDEN_DIM, 2*HIDDEN_DIM});
        p_zlbias = model.add_parameters({2*HIDDEN_DIM});

        p_rh2zr = model.add_parameters({2*HIDDEN_DIM, 2*HIDDEN_DIM});
        p_th2zr = model.add_parameters({2*HIDDEN_DIM, 2*HIDDEN_DIM});
        p_zrbias = model.add_parameters({2*HIDDEN_DIM});

        p_lrh2zlr = model.add_parameters({2*HIDDEN_DIM, 2*HIDDEN_DIM});
        p_th2zlr = model.add_parameters({2*HIDDEN_DIM, 2*HIDDEN_DIM});
        p_zlrbias = model.add_parameters({2*HIDDEN_DIM});

        p_final_lrR = model.add_parameters({3, 2*HIDDEN_DIM});

        for(auto& it : pretrained){
	    p_word.initialize(it.first, it.second);
        }
    }
 
    /*
    input:
        inst: input instance
        cg: computational graph
        num_correct: correct label counting
        pred: predict label
    output:
        return Expression of total loss
    */
    Expression BuildGraph(Instance& inst, // training instance
        ComputationGraph& cg, // computational graph
        float& num_correct,  // the number of correct prediction
        float* pred, // the number of prediction
        bool train //indication of training or not
        ) {
        const vector<unsigned>& sent = inst.words; // input words
	int label = inst.label; // sentiment label
        unsigned ts = inst.ts; // target start index
	unsigned te = inst.te; // target end index
	const unsigned slen = sent.size() ; // the length of input sentence

        l2rbuilder.new_graph(cg);  // reset builder for new graph
        l2rbuilder.start_new_sequence(); // start a new sequence (initialization of LSTM)

        r2lbuilder.new_graph(cg);  // reset builder for new graph
        r2lbuilder.start_new_sequence(); // start a new sequence (initialization of LSTM)

        // produce expressions, each experssion is associate with a parameter.
        // The experessions are used to construct computational graph
	Expression i_bias = parameter(cg, p_bias);
        
	Expression i_tag2label = parameter(cg, p_tag2label);
	Expression i_lbias = parameter(cg, p_lbias);

	Expression word_start = parameter(cg, p_start);
        Expression word_end = parameter(cg, p_end);

	Expression attbias = parameter(cg, p_attbias);
	Expression input2att = parameter(cg, p_input2att);
	Expression target2att = parameter(cg, p_target2att);
	Expression att2attexp = parameter(cg, p_att2attexp);

	Expression attbias_l = parameter(cg, p_attbias_l);
        Expression input2att_l = parameter(cg, p_input2att_l);
        Expression target2att_l = parameter(cg, p_target2att_l);
        Expression att2attexp_l = parameter(cg, p_att2attexp_l);

	Expression attbias_r = parameter(cg, p_attbias_r);
        Expression input2att_r = parameter(cg, p_input2att_r);
        Expression target2att_r = parameter(cg, p_target2att_r);
        Expression att2attexp_r = parameter(cg, p_att2attexp_r);

	Expression i_lh2zl = parameter(cg, p_lh2zl);
        Expression i_th2zl = parameter(cg, p_th2zl);
        Expression i_zlbias = parameter(cg, p_zlbias);

        Expression i_rh2zr = parameter(cg, p_rh2zr);
        Expression i_th2zr = parameter(cg, p_th2zr);
        Expression i_zrbias = parameter(cg, p_zrbias);

        Expression i_lrh2zlr = parameter(cg, p_lrh2zlr);
        Expression i_th2zlr = parameter(cg, p_th2zlr);
        Expression i_zlrbias = parameter(cg, p_zlrbias);

        Expression i_final_lrR = parameter(cg, p_final_lrR);


if(DEBUG)	cerr<<"sent size " << slen<<"\n";
        vector<Expression> i_words(slen);
        for (unsigned t = 0; t < slen; ++t) {
            i_words[t] = lookup(cg, p_word, sent[t]); // get word embeddings
            // if train, dropout is applied, which randomly set some dimensions of word embeddings to zero under the probability of dropout_p
            if (train) i_words[t] = dropout(i_words[t], pdrop); 
        }

if(DEBUG)	cerr<<"all input expression done\n";
	
	vector<Expression> l2r(slen); // store the experssion of forward lstm output in each step
	vector<Expression> r2l(slen); // store the expression of backword lstm output in each step
        Expression l2r_s = l2rbuilder.add_input(word_start); // add padding for sentence start
        Expression r2l_e = r2lbuilder.add_input(word_end); // add padding for sentence end, (backward)
        for (unsigned t = 0; t < slen; ++t) {
	    l2r[t] = l2rbuilder.add_input(i_words[t]); // add each words
            r2l[slen - 1 - t] = r2lbuilder.add_input(i_words[slen - 1 - t]);
        }
	Expression l2r_e = l2rbuilder.add_input(word_end);
        Expression r2l_s = r2lbuilder.add_input(word_start);
	

	vector<Expression> input_l; // store word experssion of leftside phrase of target
	input_l.push_back(concatenate({l2r_s,r2l_s}));
	for (unsigned t = 0; t < ts; ++t) {
		input_l.push_back(concatenate({l2r[t],r2l[t]}));
	}
	vector<Expression> input_r; // store word experssion of right phrase of target
	for (unsigned t = te+1; t < slen; ++t) {
		input_r.push_back(concatenate({l2r[t],r2l[t]}));
	}
	input_r.push_back(concatenate({l2r_e,r2l_e}));

	vector<Expression> input; // store the whole sentence experssion except target
        for (unsigned t = 0; t < slen; ++t) {
		if(t >= ts && t <= te) continue;
                input.push_back(concatenate({l2r[t],r2l[t]}));
        }
	vector<Expression> targets; // store target experssion
	for (unsigned t = ts; t <= te; ++t){
	    targets.push_back(concatenate({l2r[t],r2l[t]}));
	}
	Expression target = average(targets);


    // no word to explain, this is for attention details
    // the whole sentence attention
	vector<Expression> att(input.size());
        for(unsigned t = 0; t < input.size(); t ++){
        	att[t] = tanh(affine_transform({attbias, input2att, input[t], target2att, target}));
      	}
      	Expression att_col = transpose(concatenate_cols(att));
      	Expression attexp = softmax(att_col * att2attexp);
/*
	vector<float> weight = as_vector(cg.incremental_forward(attexp));
	for (unsigned t = 0; t < weight.size(); t ++){
		cerr<<weight[t]<<" ";
	}
	cerr<<"\n";
*/
      	Expression input_col = concatenate_cols(input);
      	Expression att_pool = input_col * attexp;
	

        // the leftside target attention
	vector<Expression> att_l(input_l.size());
        for(unsigned t = 0; t < input_l.size(); t ++){
                att_l[t] = tanh(affine_transform({attbias_l, input2att_l, input_l[t], target2att_l, target}));
        }
        Expression att_col_l = transpose(concatenate_cols(att_l));
        Expression attexp_l = softmax(att_col_l * att2attexp_l);

        Expression input_col_l = concatenate_cols(input_l);
        Expression att_pool_l = input_col_l * attexp_l;
/*
	vector<float> weight_l = as_vector(cg.incremental_forward(attexp_l));
        for (unsigned t = 0; t < weight_l.size(); t ++){
                cerr<<weight_l[t]<<" ";
        }
        cerr<<"\n";
*/
            // the rightside target attention
	vector<Expression> att_r(input_r.size());
        for(unsigned t = 0; t < input_r.size(); t ++){
                att_r[t] = tanh(affine_transform({attbias_r, input2att_r, input_r[t], target2att_r, target}));
        }
        Expression att_col_r = transpose(concatenate_cols(att_r));
        Expression attexp_r = softmax(att_col_r * att2attexp_r);

        Expression input_col_r = concatenate_cols(input_r);
        Expression att_pool_r = input_col_r * attexp_r;
/*
	vector<float> weight_r = as_vector(cg.incremental_forward(attexp_r));
        for (unsigned t = 0; t < weight_r.size(); t ++){
                cerr<<weight_r[t]<<" ";
        }
        cerr<<"\n";
*/
        // three-way gated strategy
	Expression zl = exp(affine_transform({i_zlbias, i_lh2zl, att_pool_l, i_th2zl, target}));
        Expression zr = exp(affine_transform({i_zrbias, i_rh2zr, att_pool_r, i_th2zr, target}));
        Expression zlr = exp(affine_transform({i_zlrbias, i_lrh2zlr, att_pool, i_th2zlr, target}));

        Expression zsum = sum({zl, zr, zlr});

        Expression zlgate = cdiv(zl, zsum);
        Expression zrgate = cdiv(zr, zsum);
        Expression zlrgate = cdiv(zlr,zsum);
/*
	vector<float> weight_zlr = as_vector(cg.incremental_forward(zlrgate));
        for (unsigned t = 0; t < weight_zlr.size(); t ++){
                cerr<<weight_zlr[t]<<" ";
        }
        cerr<<"\n";
	
	vector<float> weight_zl = as_vector(cg.incremental_forward(zlgate));
        for (unsigned t = 0; t < weight_zl.size(); t ++){
                cerr<<weight_zl[t]<<" ";
        }
        cerr<<"\n";

	vector<float> weight_zr = as_vector(cg.incremental_forward(zrgate));
        for (unsigned t = 0; t < weight_zr.size(); t ++){
                cerr<<weight_zr[t]<<" ";
        }
        cerr<<"\n";
*/
        // final representation of sentence with the specific target
        Expression final_lr = cwise_multiply(zlgate, att_pool_l) + cwise_multiply(zrgate, att_pool_r) + cwise_multiply(zlrgate, att_pool);

/*
if(DEBUG)	cerr<<"bilstm done\n";
	Expression i_r_t = tanh(i_bias + i_final_lrR * final_lr);
        Expression i_lstm_t = 2.0 *tanh(i_lbias + i_tag2label * i_r_t);//add tanh
        Expression total_score = i_lstm_t;
if(DEBUG)	cerr<<"sentence bias done\n";
        
        auto prob_value = as_scalar(cg.incremental_forward(total_score));
if(DEBUG)   cerr<<"prob_value "<<prob_value<<" gold label "<<label<<"\n";
	if(prob_value >= THRESHOLD) { if(pred) (*pred) = 1; if((int)label == 1) num_correct += 1;}
        else if(prob_value <= -THRESHOLD) { if(pred) (*pred) = -1; if((int)label == -1) num_correct += 1;}
        else {if(pred) (*pred) = 0; if((int)label == 0) num_correct += 1;}
        Expression goldy = input(cg, label);
	Expression output_loss = squared_distance(total_score, goldy);

	return output_loss;
*/
	Expression i_r_t = tanh(i_bias + i_final_lrR * final_lr);
	Expression output_loss = pickneglogsoftmax(i_r_t, label+1);
 
	auto prob_value = as_vector(cg.incremental_forward(i_r_t));
	float best = prob_value[0];
	unsigned bestk = 0;
	for(unsigned i = 1; i < prob_value.size(); i ++){
		if(best < prob_value[i]){best = prob_value[i]; bestk = i;}
	}
	if(bestk == label+1) num_correct += 1;

	if(pred){
	if(bestk == 0) (*pred) = -1;
	else if(bestk == 1) (*pred) = 0;
	else (*pred) = 1;
	}
	return output_loss;
    }
};
float getF1(float cor, // the number of correct label
    float pred, // the number of predicted label
    unsigned gold // the number of gold label
    ){
        if(cor == 0) return 0;
        float P = cor / pred;
        float R = cor / gold;
        return P*R*2/(P+R);
}
void output(vector<Instance>& instances, // instances
    LSTMClassifier& lstmClassifier, // LSTM classifier
    const unsigned& neu, // the total number of neutron label
    const unsigned& pos, // the total number of positive label
    const unsigned& neg // the total number of negative label
    )
{   
	int cnt = 0;
    float num_correct = 0;
    float loss = 0;
    float pred, acc, f1;
    float pred_neu, pred_pos, pred_neg, cor_neu, cor_pos, cor_neg; 
    pred_neu = pred_pos = pred_neg = cor_neu = cor_pos = cor_neg = 0;
    ofstream out("OUT.txt");
    for (auto& sent : instances) {
	cerr <<cnt++<<"\n";
        ComputationGraph cg;
        Expression nll = lstmClassifier.BuildGraph(sent, cg, num_correct, &pred, false);
        loss += as_scalar(cg.incremental_forward(nll)); 
        if(pred == 0) {out << 0 << "\n"; pred_neu += 1; if(pred == sent.label) cor_neu +=1;}
        else if(pred == 1) {out << 1 << "\n"; pred_pos += 1; if(pred == sent.label) cor_pos += 1;}
        else if(pred == -1) {out << -1 << "\n";pred_neg += 1; if(pred == sent.label) cor_neg += 1;}
    }
    out.flush();
    out.close();
    acc = num_correct/ instances.size();
    cerr<<"Loss:"<< loss/ instances.size() << " ";
    cerr<<"Accuracy:"<< num_correct <<"/" << instances.size() <<" "<<acc<<" ";
    
    float f1_neu = getF1(cor_neu, pred_neu, neu);
    float f1_pos = getF1(cor_pos, pred_pos, pos);
    float f1_neg = getF1(cor_neg, pred_neg, neg);
    
    f1 = (f1_neu + f1_pos + f1_neg)/3;
    
    cerr<<"Macro F1: "<< f1 <<" ";
}
/*
evaluate the model
input:
    instances: the set of instances waiting to be evaluated
    lstmClassifier: the model
    acc: accuracy
    f1: F1 score
    neu: the number of neutron label
    pos: the number of positive label
    neg: the number of negative label
output:
    none
*/
void evaluate(vector<Instance>& instances,// instances
    LSTMClassifier& lstmClassifier, // LSTM classifier
    float& acc, // accuracy
    float& f1, // f-score
    const unsigned& neu, // the total number of neutron label
    const unsigned& pos, // the total number of positive label
    const unsigned& neg // the total number of negative label
    )
{
    float num_correct = 0;
    float loss = 0;
    float pred;
    float pred_neu, pred_pos, pred_neg, cor_neu, cor_pos, cor_neg;
    pred_neu = pred_pos = pred_neg = cor_neu = cor_pos = cor_neg = 0;
    for (auto& sent : instances) {
        ComputationGraph cg;
        Expression nll = lstmClassifier.BuildGraph(sent, cg, num_correct, &pred, false);
        loss += as_scalar(cg.incremental_forward(nll));
        if(pred == 0) {pred_neu += 1; if(pred == sent.label) cor_neu +=1;}
        else if(pred == 1) {pred_pos += 1; if(pred == sent.label) cor_pos += 1;}
        else if(pred == -1) {pred_neg += 1; if(pred == sent.label) cor_neg += 1;}
    }
    acc = num_correct/ instances.size();
    cerr<<"Loss:"<< loss/ instances.size() << " ";
    cerr<<"Accuracy:"<< num_correct <<"/" << instances.size() <<" "<<acc<<" ";

    float f1_neu = getF1(cor_neu, pred_neu, neu);
    float f1_pos = getF1(cor_pos, pred_pos, pos);
    float f1_neg = getF1(cor_neg, pred_neg, neg);

    f1 = (f1_neu + f1_pos + f1_neg)/3;
    cerr<<"Macro F1: "<< f1 <<" ";
}

/*
    main entry
*/
int main(int argc, char** argv) {
    /*
    initailized dynet using the input parameters, which are stored in argv
    */
    DynetParams dynet_params = extract_dynet_params(argc, argv); //extract dynet parameter from argv
    dynet_params.random_seed = 1989121013; // init random seed
    dynet::initialize(dynet_params); // init dynet
  
    /*
    print out the input parameters
    */
    cerr << "COMMAND:";
    for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
    cerr << endl;

    /*
    save the input parameters into conf, which is used to initialied hyper parameters of neural network
    */
    po::variables_map conf;
    InitCommandLine(argc, argv, &conf);
    WORD_DIM = conf["word_dim"].as<unsigned>();
    HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
    TAG_HIDDEN_DIM = conf["tag_hidden_dim"].as<unsigned>();
    LAYERS = conf["layers"].as<unsigned>();
    unk_prob = conf["unk_prob"].as<float>();
    pdrop = conf["pdrop"].as<float>();

    DEBUG = conf.count("debug");

    assert(unk_prob >= 0.); assert(unk_prob <= 1.);
    assert(pdrop >= 0.); assert(pdrop <= 1.);

    /*
    training, development and test instance
    */
    vector<Instance> training,dev,test;
    string line;
    
    kUNK = wd.convert("*UNK*");
    //reading pretrained
    if(conf.count("pretrained")){
      cerr << "Loading from " << conf["pretrained"].as<string>() << " as pretrained embedding with" << WORD_DIM << " dimensions ... ";
      ifstream in(conf["pretrained"].as<string>().c_str());
      string word;
      unk_embedding.resize(WORD_DIM, 0);
      while(in>>word){
        vector<float> v(WORD_DIM);
        for(unsigned i = 0; i < WORD_DIM; i++) {in>>v[i]; unk_embedding[i] += v[i];}
        pretrained[wd.convert(word)] = v;
      }
      for(unsigned i = 0; i < WORD_DIM; i ++) unk_embedding[i] /= pretrained.size();
      cerr << pretrained.size() << " ok\n";
    }
    
    /*
    reading the lexicon, actually these are not used in the whole program.
    this reading is to keep the lexicon index consistent if you use the pretrained model for testing.
    */
    if(conf.count("lexicon")){
      cerr << "Loading from " << conf["lexicon"].as<string>() << " as lexion dictionary ...";
      ifstream in(conf["lexicon"].as<string>().c_str());
      string word;
      float v;
      while(in>>word){
    	in>>v;
	lex_dict[wd.convert(word)] = v;
      }
      cerr << lex_dict.size() << " ok\n";
    }
    train_neu = train_pos = train_neg = 0;

    //reading training data
    cerr << "Loading from " << conf["training_data"].as<string>() << "as training data : ";
    {
      ifstream in(conf["training_data"].as<string>().c_str());
      assert(in);
      while(getline(in, line)) {
        Instance instance;
        instance.load(line, train_neu, train_pos, train_neg);
        training.push_back(instance);
      }
      cerr<<training.size()<<" where neu: "<<train_neu<<" pos: "<<train_pos<<" neg: "<<train_neg<<"\n";
    }

    //couting
    set<unsigned> training_vocab; //the set of indexs of words which are in training data
    set<unsigned> singletons; // the set of indexs of words which appear only once in training data
    {
      map<unsigned, unsigned> counts;
      for (auto& sent : training){
	const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<unsigned>& words = sent.words;
	for (unsigned i = 0; i < sent.size(); ++i){
	  if(pretrained.size() > 0){
	    if(pretrained.count(raws[i])) words[i] = raws[i];
	    else if(pretrained.count(lows[i])) words[i] = lows[i];
	  }
          training_vocab.insert(words[i]); counts[words[i]]++;
	}
      }
      for (auto wc : counts)
        if (wc.second == 1) singletons.insert(wc.first);
      
      cerr<<"the training word dict size is " << training_vocab.size()
	     << " where The singletons have " << singletons.size() << "\n";
    }

    //replace unk, uniformly randomly replace the singletons with UNK 
    {
      int unk = 0;
      int total = 0;
      for(auto& sent : training){
        for(auto& w : sent.words){
          if(singletons.count(w) && dynet::rand01() < unk_prob){
	  	w = kUNK;
		unk += 1;
 	  }
          total += 1;
        }
      }
      cerr << "the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
    }

    //import lexicon score for each word
    //these are not used
    {
      for(auto& sent : training){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
	vector<float>& lex_score = sent.lex_score;
	for(unsigned i = 0; i < sent.size(); ++i){
	  if(lex_dict.count(raws[i])) lex_score[i] = lex_dict[raws[i]];
	  else if(lex_dict.count(lows[i])) lex_score[i] = lex_dict[lows[i]];
	  else lex_score[i] = noscore;
        }
      }
    }
    dev_neu = dev_pos = dev_neg = 0;
    //reading dev data 
    if(conf.count("dev_data")){
      cerr << "Loading from " << conf["dev_data"].as<string>() << "as dev data : ";
      ifstream in(conf["dev_data"].as<string>().c_str());
      string line;
      while(getline(in,line)){
        Instance inst;
        inst.load(line, dev_neu, dev_pos, dev_neg);
        dev.push_back(inst);
      }
      cerr<<dev.size()<<" where neu: "<<dev_neu<<" pos: "<<dev_pos<<" neg: "<<dev_neg<<"\n";
    }

    //replace word with unk
    {
      int unk = 0;
      int total = 0;
      for(auto& sent : dev){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<unsigned>& words = sent.words;
	for(unsigned i = 0; i < sent.size(); i ++){
          if(pretrained.count(raws[i])) words[i] = raws[i];
	  else if(pretrained.count(lows[i])) words[i] = lows[i];
	  else if(training_vocab.count(raws[i])) words[i] = raws[i];
	  else{
	  	words[i] = kUNK;
		unk += 1;
	  }
          total += 1;
        }
      }
      cerr << "the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
    }
 
    {
      for(auto& sent : dev){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<float>& lex_score = sent.lex_score;
        for(unsigned i = 0; i < sent.size(); ++i){
          if(lex_dict.count(raws[i])) lex_score[i] = lex_dict[raws[i]];
          else if(lex_dict.count(lows[i])) lex_score[i] = lex_dict[lows[i]];
          else lex_score[i] = noscore;
        }
      }
    }
    tst_neu = tst_pos = tst_neg = 0;
    //reading test data
    if(conf.count("test_data")){
      cerr << "Loading from " << conf["test_data"].as<string>() << "as test data : ";
      ifstream in(conf["test_data"].as<string>().c_str());
      string line;
      while(getline(in,line)){
        Instance inst;
        inst.load(line, tst_neu, tst_pos, tst_neg);
        test.push_back(inst);
      }
      cerr<<test.size()<<" where neu: "<<tst_neu<<" pos: "<<tst_pos<<" neg: "<<tst_neg<<"\n";
    }

    //replace word with unk
    {
      int unk = 0;
      int total = 0;
      for(auto& sent : test){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<unsigned>& words = sent.words;
        for(unsigned i = 0; i < sent.size(); i ++){
          if(pretrained.count(raws[i])) words[i] = raws[i];
          else if(pretrained.count(lows[i])) words[i] = lows[i];
          else if(training_vocab.count(raws[i])) words[i] = raws[i];
          else{
                words[i] = kUNK;
                unk += 1;
          }
          total += 1;
        }
      }
      cerr << "the number of word is: "<< total << ", where UNK is: "<<unk<<"("<<unk*1.0/total<<")\n";
    }

    {
      for(auto&sent : test){
        const vector<unsigned>& raws = sent.raws;
        const vector<unsigned>& lows = sent.lows;
        vector<float>& lex_score = sent.lex_score;
        for(unsigned i = 0; i < sent.size(); ++i){
          if(lex_dict.count(raws[i])) lex_score[i] = lex_dict[raws[i]];
          else if(lex_dict.count(lows[i])) lex_score[i] = lex_dict[lows[i]];
          else lex_score[i] = noscore;
        }
      }
    }

    VOCAB_SIZE = wd.size();

    // construct model path
    ostringstream os;
    os << "lstmclassifier"
       << '_' << WORD_DIM
       << '_' << HIDDEN_DIM
       << '_' << LAYERS
       << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameter will be written to: " << fname << endl;

    float best = 0;
    float bestf1 = 0;

    Model model; // model instance
    Trainer* sgd = nullptr; // trainer instance

    unsigned method = conf["train_methods"].as<unsigned>();
    if(method == 0)
  	sgd = new SimpleSGDTrainer(&model,0.1, 0.1);
    else if(method == 1)
	sgd = new MomentumSGDTrainer(&model,0.01, 0.9, 0.1);
    else if(method == 2){
	sgd = new AdagradTrainer(&model);
	sgd->clipping_enabled = false;	
    }
    else if(method == 3){
	sgd = new AdamTrainer(&model);
  	sgd->clipping_enabled = false;
    } 
    LSTMClassifier lstmClassifier(model); // sentiment classifier

	if (conf.count("model")) { // if pretrained model is given
    string fname = conf["model"].as<string>();
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
    }
if(conf.count("test") == 0){
if(DEBUG)	cerr<<"begin\n";
    unsigned report_every_i = conf["report_i"].as<unsigned>();
    unsigned dev_report_every_i = conf["dev_report_i"].as<unsigned>();
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int exceed_count = 0;
    unsigned count = 0;
    while(count < conf["count_limit"].as<unsigned>()) { // train start
        Timer iteration("completed in");
        float loss = 0;
        unsigned ttags = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training.size()) { // each epoch, we test the model on development dataset
                si = 0;
                if (first) {
                    first = false;
                }
                else {
                    sgd->update_epoch();
                    if (1) {
                        float acc = 0.f;
			float f1 = 0.f;
                        cerr << "\n***DEV [epoch=" << (lines / (float)training.size()) << "] ";
                        evaluate(dev, lstmClassifier, acc, f1, dev_neu, dev_pos, dev_neg);
                        if (acc > best || (fabs(acc - best) <= 0.0000001 && f1 > bestf1)) { // if get better model on development dataset, we test the model on test dataset
                            best = acc;
			    bestf1 = f1;
                            cerr<< "Exceed" << " ";
                            float tacc = 0;
			    float tf1 = 0;
                            evaluate(test, lstmClassifier, tacc, tf1, tst_neu, tst_pos, tst_neg);
                            ofstream out(fname);
                            boost::archive::text_oarchive oa(out);
                            oa << model;
                            exceed_count ++;
                        }
			cerr<<"\n";
                    }
                }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
                count++;
            }

            ComputationGraph cg;
            auto& sentx_y = training[order[si]];
	    float num_correct = 0;
            Expression nll= lstmClassifier.BuildGraph(sentx_y, cg, num_correct, NULL, true);
	    loss += as_scalar(cg.incremental_forward(nll));
            cg.backward(nll);
            sgd->update(1.0);
            ++si;
            ++lines;
            ++ttags;
        }
        sgd->status();
        cerr << " E = " << (loss / ttags) <<" "<<loss << "/"<<ttags<<" ";

        report++;
        continue;
        if ( report % dev_report_every_i == 1 ) {
            float acc = 0.f;
	    float f1 = 0.f;
            cerr << "\n***DEV [epoch=" << (lines / (float)training.size()) << "] ";
            evaluate(dev, lstmClassifier, acc, f1, dev_neu, dev_pos, dev_neg);
            if (acc > best || (fabs(acc - best) <= 0.0000001 && f1 > bestf1)) {
                best = acc;
		bestf1 = f1;
                cerr<< "Exceed" << " ";
                float tacc = 0;
		float tf1 = 0;
                evaluate(test, lstmClassifier, tacc, tf1, tst_neu, tst_pos, tst_neg);
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
                exceed_count++;
            }
	    cerr<<"\n";
        }
    }
    delete sgd;
	}
    else{
        output(test, lstmClassifier, tst_neu, tst_pos, tst_neg);
    }
}

