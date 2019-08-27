/*
 * ECPF.java
 * author: Robert William Anderson - The University of Auckland
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. See the License for the specific
 * language governing permissions and limitations under the
 * License.
 * 
 */

package moa.classifiers.meta;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.TreeSet;

import org.apache.commons.math3.util.FastMath;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.bayes.NaiveBayesOpen;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.core.GaussianEstimator;
import moa.core.InstanceExample;
import moa.core.Utils;
import moa.evaluation.RobWindowAUC;
import moa.evaluation.RobWindowAUC.Estimator.Score;
import moa.options.FlagOption;
import moa.streams.ArffFileStream;

/*Not for public use - provided as part of submission for CIKM 2019 by Robert Anderson*/
public class GCPF extends AbstractCPF{

	private static final long serialVersionUID = 1L;

	public IntOption modelCheckFreqOption = new IntOption(
            "modelCheckFreq",
            'c',
            "Frequency of comparing models (new vs. reused) to decide which to use for classifying incoming instances",
            1, 0, 100000);
    
    public FlagOption trackConceptSimOption = new FlagOption(
            "trackConceptSim",
            's',
            "A flag that stores all classifiers so classifier similarity can be compared post-hoc");
	
	//no max on classifiers, size restricted by fade points
	ArrayList<Classifier> classifierCollection = new ArrayList<Classifier>();
	ArrayList<Classifier> removedClassifierCollection = new ArrayList<Classifier>();
	ArrayList<String> classifierDestination = new ArrayList<String>(); //tracks what ends up happening to each classifier
	ArrayList<Integer> classifierStart = new ArrayList<Integer>(); //tracks when each classifier starts
	Integer currentClassifier = 0;
	Classifier newModel = null;

	//buffer for instances
	Instances buffer = new Instances();
	
	double similarityMargin;
	int bufferSize;
	int fadePoints;
	int modelCheckFreq;
    int ddmPriorLevel = 0;
    
    //counters for measuring ECPF behaviour
    int numberInstances = 0;
    int totalBufferInstances = 0;
    public int numDrifts = 0;
    public int modelReuses = 0;
    public int modelMerges = 0;
    int maxModels = 0;
    int currentModels = 0;
    int reuseFlag = 1;
    
    //Counters for classifier
	int currMinCorrect = 0;
	int currMajCorrect = 0;
	int newMinCorrect = 0;
	int newMajCorrect = 0;
	int totalMinInst = 0;
	int totalMajInst = 0;
	
	boolean verbose = false;
    
	// Object to hold measurements relating to model - accuracy and model similarity
	ArrayList<Integer[]> modelAccuracyMeasurements = new ArrayList<Integer[]>();
	ArrayList<HashMap<Integer, Integer[]>> modelComparisonMeasurements  = new ArrayList<HashMap<Integer, Integer[]>>();;
	
    //objects for model fading
    boolean fadeModels;
    boolean trackConceptSim;
    HashMap<Integer, Integer> modelFadeScores = new HashMap<Integer, Integer>();
    int modelsFaded = 0;
    
    //objects for adaptive AUC
    RobWindowAUC eval = new RobWindowAUC();
	double decisionBoundary = 0.5;
    
    public static final int DDM_BUILD_BUFFER = 3;
    
	@Override
	public void resetLearningImpl() {
		
		this.classifierCollection.clear();
		this.removedClassifierCollection.clear();
		this.modelAccuracyMeasurements.clear();
		this.modelComparisonMeasurements.clear();
		this.classifierDestination.clear();
		this.modelFadeScores.clear();
		
		
		this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
		
		this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
		
		this.newModel = null;
		this.similarityMargin = this.similarityBetweenModelsOnBufferOption.getValue();
		this.fadePoints = this.fadePointsOption.getValue();
		this.modelCheckFreq = this.modelCheckFreqOption.getValue();
		this.buffer.delete();
	    this.numberInstances = 0;
	    this.totalBufferInstances = 0;
	    this.modelsFaded = 0;
	    this.numDrifts = 0;
	    this.modelReuses = 0;
	    this.modelMerges = 0;
	    this.maxModels = 0;
		this.currentModels = 0;
		this.currMinCorrect = 0;
		this.currMajCorrect = 0;
		this.newMinCorrect = 0;
		this.newMajCorrect = 0;
		this.totalMinInst = 0;
		this.totalMajInst = 0;
		
	   //model management flags
	    fadeModels = fadeModelOption.isSet() ? true : false;
	    trackConceptSim = trackConceptSimOption.isSet() ? true : false;

	    
	    addModel(((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy());
	    this.classifierCollection.get(currentClassifier).prepareForUse();
	 
	}
	
	private void addModel(Classifier newModel){
		this.currentModels = this.currentModels + 1;
		this.currentClassifier = classifierCollection.size();
	    this.classifierCollection.add(newModel);
	    this.modelAccuracyMeasurements.add(new Integer[] {0,0,0,0});
	    this.modelComparisonMeasurements.add(new HashMap<Integer, Integer[]>());
	    if(fadeModels) modelFadeScores.put(currentClassifier, 0);
	    classifierDestination.add(Integer.toString(classifierCollection.size() - 1));
	    classifierStart.add(numberInstances);
	}
	
    //Run usual drift-detection but check for equivalent models on change
    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	
    	this.numberInstances++;
        boolean prediction = getPrediction(inst);
        //if(inst.classValue() == 1.0)
        this.driftDetectionMethod.input(prediction ? 0.0 : 1.0);
    	
        this.ddmLevel = DDM_INCONTROL_LEVEL;
        if (this.driftDetectionMethod.getChange()) {
            this.ddmLevel = DDM_OUTCONTROL_LEVEL;
        }
        if (this.driftDetectionMethod.getWarningZone()) {
            this.ddmLevel = DDM_WARNING_LEVEL;
        }

        
        switch (this.ddmLevel) {
            case DDM_WARNING_LEVEL:
            	if(this.ddmLevel != this.ddmPriorLevel){
            		this.warningDetected++;
            		buffer.delete();
            	}
                buffer.add(inst);
                break;
            case DDM_OUTCONTROL_LEVEL:
            	if(verbose) System.out.println("Drift detected at inst " + numberInstances);
            	buffer.add(inst);
                this.changeDetected++;
                numDrifts++;
                modelReuses++;
                compareClassifiers();
                this.getNextModel();
                getNewBoundary();
                eval.reset();
                break;

            case DDM_INCONTROL_LEVEL:
            	//System.out.println("DDM_INCONTROL_LEVEL");
            	trainClassifiers(inst);
                break;
            	
            default:
            	System.out.println("ERROR!");

        }
        ddmPriorLevel = ddmLevel;
    }
    
    private boolean getPrediction(Instance inst){
    	
    	int trueClass = (int) inst.classValue();
    	double[] votes;
    	double pred;
    	
    	if(newModel != null){
    		votes = this.newModel.getVotesForInstance(inst);
    		if(votes.length > 1)
    			pred = votes[1]/votes[0]+votes[1] > decisionBoundary ? 1.0 : 0.0; 
    		else pred = 0;
            
            boolean newPred = pred == trueClass;
    		if(newPred){
    			if(inst.classValue() == 0) newMajCorrect ++;
    		} else{
    			if(inst.classValue() == 0) newMinCorrect ++;
    		}
    	}
    	votes = classifierCollection.get(currentClassifier).getVotesForInstance(inst);

    	 
    	if(votes.length > 1)
			pred = votes[1]/votes[0]+votes[1] > decisionBoundary ? 1.0 : 0.0; 
		else pred = 0;
    	boolean currPred = pred == trueClass;
    	if (currPred){
			if(inst.classValue() == 0) currMajCorrect ++;
		} else{
			if(inst.classValue() == 0) currMinCorrect ++;
		}
    	
    	eval.addResult(new InstanceExample(inst), votes);
    	if (inst.classValue() == 1) totalMinInst++;
    	else totalMajInst++;
    	
    	if(totalMinInst + totalMajInst % modelCheckFreq == 0)
    		compareClassifiers();
    		
		return currPred;
    }
    
    private void trainClassifiers(Instance inst){
    	((Classifier)this.classifierCollection.get(currentClassifier)).trainOnInstance(inst);
    	if(newModel != null) newModel.trainOnInstance(inst);
    }
    
    private void compareClassifiers(){
    	if(getGMean(totalMinInst, currMinCorrect, totalMajInst, currMajCorrect)  < getGMean(totalMinInst, newMinCorrect, totalMajInst, newMajCorrect)){
    		int tempMinCorrect = currMinCorrect;
    		int tempMajCorrect = currMajCorrect;
    		currMinCorrect = newMinCorrect;
    		currMajCorrect = newMajCorrect;
    		Classifier temp = classifierCollection.get(currentClassifier);
    		classifierCollection.set(currentClassifier, newModel);
    		newModel = temp;
    		newMinCorrect = tempMinCorrect;
    		newMajCorrect = tempMajCorrect;
    		reuseFlag = reuseFlag * -1;
    	}
    }
    
    private void getNextModel(){
    	
    	//Update accuracy measurements with winning model
        this.modelAccuracyMeasurements.get(currentClassifier)[0] = this.modelAccuracyMeasurements.get(currentClassifier)[0] + totalMinInst;
        this.modelAccuracyMeasurements.get(currentClassifier)[1] = this.modelAccuracyMeasurements.get(currentClassifier)[1] + currMinCorrect;
        this.modelAccuracyMeasurements.get(currentClassifier)[2] = this.modelAccuracyMeasurements.get(currentClassifier)[2] + totalMajInst;
        this.modelAccuracyMeasurements.get(currentClassifier)[3] = this.modelAccuracyMeasurements.get(currentClassifier)[3] + currMajCorrect;
        
    	this.ddmLevel =  DDM_OUTCONTROL_LEVEL;
    	totalBufferInstances += buffer.size();
    	currentClassifier = null;
    	
    	//get results per model on this comparison window
    	ArrayList<BitSet> thisBufferResults = new ArrayList<BitSet>();
    	
    	//get indices of current models
    	ArrayList<Integer> currentModels = new ArrayList<Integer>();
    	for(int i = 0; i < classifierCollection.size(); i++){
    		if(!(classifierCollection.get(i) == null))
    			currentModels.add(i);
    	}
    	
    	if(verbose) System.out.print(", buffer size:" + buffer.size());
    	
    	for(int i = 0; i < currentModels.size(); i++){
    		thisBufferResults.add(new BitSet(buffer.size()));
        	
    		for(int j = 0; j < buffer.size(); j++){
    			if (!((Classifier) classifierCollection.get(currentModels.get(i))).correctlyClassifies(buffer.get(j)))
    				thisBufferResults.get(i).set(j);
    		}
    	}
	    
    	for(int i = 0; i < currentModels.size(); i++){
    		for(int j = i + 1; j > i & j < currentModels.size(); j++){
    			
    			//ensure comparison tracking values have been initialised
    			if(modelComparisonMeasurements.get(currentModels.get(i)).get(currentModels.get(j)) == null)
    				modelComparisonMeasurements.get(currentModels.get(i)).put(currentModels.get(j), 
    						new Integer[]{0,0});
    			    			
    			BitSet difference = (BitSet) thisBufferResults.get(j).clone();
    			difference.xor(thisBufferResults.get(i)); 			
    			int seen_before = modelComparisonMeasurements.get(currentModels.get(i)).get(currentModels.get(j))[0];
    			int agreed_before = modelComparisonMeasurements.get(currentModels.get(i)).get(currentModels.get(j))[1];
    			int seen_this_buffer = buffer.size();
    			int agreed_this_buffer = buffer.size() - difference.cardinality();
    			
    			modelComparisonMeasurements.get(currentModels.get(i)).get(currentModels.get(j))[0] = seen_before + seen_this_buffer;
    			modelComparisonMeasurements.get(currentModels.get(i)).get(currentModels.get(j))[1] = agreed_before + agreed_this_buffer;
    		}
    	}
    	
    	//Merge similar models and simplify model results
    	
    	ArrayList<Integer> mergedModels = mergeModels(currentModels);
    	for(int i = 0; i < mergedModels.size(); i++){
    		int thisIndex = currentModels.indexOf(mergedModels.get(i));
    		currentModels.remove(thisIndex);
    		thisBufferResults.remove(thisIndex);    		
    	}
    	
		//add a new model to contend with existing models
    	//train it on even instances in buffer
		newModel = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
		newModel.prepareForUse();
		BitSet newModelResults = new BitSet(buffer.size());
		
		//Here we have a double buffer and will initialise a new model on all warning zone instances
		//train new model
		for(int i = 0; i < buffer.size(); i++){
			newModel.trainOnInstance(buffer.get(i));
		}
		
    	//check older models to find best accuracy on buffer
    	double[] modelAccuracy = new double[thisBufferResults.size()];
    	int bestModelIndex = 0;
    	double maxAcc = 0;
		for(int i = 0; i < thisBufferResults.size(); i++){
    		modelAccuracy[i] = (double)(thisBufferResults.get(i).size() - thisBufferResults.get(i).cardinality())
    				/(double)(thisBufferResults.get(i).size());
    		if (modelAccuracy[i] > maxAcc){
    			bestModelIndex = i;
    			maxAcc = modelAccuracy[i];
    		}
    	}
		
		//Make copy of existing model to use
		addModel(classifierCollection.get(currentModels.get(bestModelIndex)).copy());
		currentModels.add(currentClassifier);

    	if (this.fadeModels) fadeModels(currentModels);
    	buffer.delete();
    	
    	this.maxModels = Math.max(this.currentModels, this.maxModels);
    	if(verbose) System.out.println("Model selected: " + currentClassifier);
    	
    	//Reset counters for next concept
		currMinCorrect = 0;
		currMajCorrect = 0;
		newMinCorrect = 0;
		newMajCorrect = 0;
		totalMinInst = 0;
		totalMajInst = 0;
    	reuseFlag = 1;
    }

    //If a model acts the same way as another model similarityMargin proportion of the time, 
    //keep the model with higher accuracy. The kept model gets fade points
	private ArrayList<Integer> mergeModels(ArrayList<Integer> currentModels){
		int modelToRemove;
		int modelToKeep;
		
		ArrayList<Integer> removedModels = new ArrayList<Integer>();
		for(int i = 0; i < currentModels.size(); i++){
			int modelA = currentModels.get(i);
			if(classifierCollection.get(modelA) == null) continue;
			for(int j = i + 1; j < currentModels.size(); j++){
				int modelB = currentModels.get(j);
				if(classifierCollection.get(modelB) == null) continue;
				if((double)(modelComparisonMeasurements.get(modelA).get(modelB)[1])/(double)(modelComparisonMeasurements.get(modelA).get(modelB)[0]) >= similarityMargin
						&& modelComparisonMeasurements.get(modelA).get(modelB)[0] >= 60){
					modelMerges++;
					if(getGMean(modelAccuracyMeasurements.get(modelA)[0], modelAccuracyMeasurements.get(modelA)[1], modelAccuracyMeasurements.get(modelA)[2], modelAccuracyMeasurements.get(modelA)[3]) >=
							getGMean(modelAccuracyMeasurements.get(modelB)[0], modelAccuracyMeasurements.get(modelB)[1], modelAccuracyMeasurements.get(modelB)[2], modelAccuracyMeasurements.get(modelB)[3])){
						modelToRemove = modelB;
						modelToKeep = modelA;
					} else {
						modelToRemove = modelA;
						modelToKeep = modelB;
					}
					
					if(verbose) System.out.println("Model " + modelToRemove + " merged with model " + modelToKeep);
					classifierDestination.set(modelToRemove, Integer.toString(modelToKeep));
					stringReplace(Integer.toString(modelToRemove), Integer.toString(modelToKeep), classifierDestination);
					removeModel(modelToRemove);
					removedModels.add(modelToRemove);
					if(fadeModels){
						modelFadeScores.put(modelToKeep, modelFadeScores.get(modelToKeep) + ((modelFadeScores.get(modelToRemove) == null) ? 0 : modelFadeScores.get(modelToRemove)));
						modelFadeScores.put(modelToRemove, null);
					}
					if(modelToRemove == currentModels.get(i)){ 
						i++;
						break;
					}
				}
			}
		}
		return removedModels;
	}
	
	private void removeModel(int modelToRemove){
		this.currentModels = this.currentModels - 1;
		if(trackConceptSim)this.removedClassifierCollection.add(classifierCollection.get(modelToRemove));
		classifierCollection.set(modelToRemove, null);
		modelComparisonMeasurements.set(modelToRemove,null);
		modelAccuracyMeasurements.set(modelToRemove,null);
	}
	
	private void fadeModels(ArrayList<Integer> currentModels){
		int score_to_add = fadePoints;
    	for(int i:currentModels){
    		if (i == currentClassifier){
    			modelFadeScores.put(currentClassifier, score_to_add 
    				+ ((modelFadeScores.get(currentClassifier) == null) ? 0 : modelFadeScores.get(currentClassifier))); //handles if model is new and not yet in modelFadeScores
    		} else if (modelFadeScores.get(i) != null){ // null is case where model has been merged
    			modelFadeScores.put(i, modelFadeScores.get(i) - 1);
    			if(modelFadeScores.get(i) == 0){
    				if(verbose) System.out.println("Model " + i + " faded");
    				removeModel(i);
    				this.modelsFaded++;
    				classifierDestination.set(i, "F");
    			}
    		}
    	}
	}

	@Override
    public double[] getVotesForInstance(Instance inst) {
        return this.classifierCollection.get(currentClassifier).getVotesForInstance(inst);
    }
	
	public boolean getWarning(){
		if(this.driftDetectionMethod.getWarningZone()) return true;
		return false;
	}
	
	public boolean getDrift(){
		if(this.driftDetectionMethod.getChange()) 
			return true;
		return false;
	}
	
	public void clearBuffer(){
		this.buffer = null;
	}
	
    //Find optimal G-mean point in scoretree by iterating through scores
	void getNewBoundary(){
		TreeSet<Score> sortedScores = eval.getAucEstimator().getScoreTree();
		double numPos = eval.getAucEstimator().getPos();
		double numNeg = eval.getAucEstimator().getNeg();

		double optimalBoundary = 1.0;
		double bestGMean = 0.0;
		double posSeen = 0;
		double negSeen = 0;
		double lastScore = 1.0;
		
		for (Score s : sortedScores){
			if(s.isPositive) posSeen++;
			else negSeen++;
			if (s.value != lastScore) {
				double thisGMean = getGMean(posSeen/numPos, 1-(negSeen/numNeg));
				//System.out.println(lastScore + "," + bestGMean + "," + thisGMean);
				if(thisGMean > bestGMean) {
					bestGMean = thisGMean;
					optimalBoundary = lastScore;
				}
				lastScore = s.value;
			}
		}
		decisionBoundary = optimalBoundary;
		
	}
	
	public double getGMean(int posSeen, int posCorrect, int negSeen, int negCorrect){
		double posAcc = (posCorrect * 1.0)/Math.max(1.0, posSeen * 1.0);
		double negAcc = (negCorrect * 1.0)/Math.max(1.0, negSeen * 1.0);
		return getGMean(posAcc, negAcc);
	}
	
	public double getGMean(double positiveAccuracy, double negativeAccuracy) {
		return FastMath.sqrt(positiveAccuracy * negativeAccuracy);
	}

	//Iterate through arraylist of strings and return arraylist with all of value a replaced with value b
	public ArrayList<String> stringReplace(String a, String b, ArrayList<String> al){
		for(int i = 0; i < al.size(); i++) if(al.get(i).equals(a)) al.set(i, b);
		return al;
	}

}
