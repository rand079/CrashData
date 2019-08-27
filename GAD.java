package moa.classifiers.meta;

import java.util.TreeSet;

import org.apache.commons.math3.util.FastMath;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ADWINChangeDetector;
import moa.classifiers.drift.DriftDetectionMethodClassifierExt;
import moa.classifiers.meta.WEKAClassifier;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.Utils;
import moa.evaluation.RobWindowAUC;
import moa.evaluation.WindowAUCImbalancedPerformanceEvaluator;
import moa.evaluation.RobWindowAUC.Estimator.Score;

/*Not for public use - provided as part of submission for CIKM 2019 by Robert Anderson*/
public class GAD extends DriftDetectionMethodClassifierExt {

	RobWindowAUC eval = new RobWindowAUC();
	double decisionBoundary = 0.5;
	public double changeDetected = 0;
	
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        //this.numberInstances++;
        int trueClass = (int) inst.classValue();
        boolean prediction;
        
		//double[] x = learner.getVotesForInstance(inst);
		double[] votes = this.classifier.getVotesForInstance(inst);
		eval.addResult(new InstanceExample(inst), votes);
		double pred;
		if(votes.length > 1)
			pred = votes[1]/votes[0]+votes[1] > decisionBoundary ? 1.0 : 0.0; 
		else pred = 0;
        
        if (pred == trueClass) {
            prediction = true;
        } else {
            prediction = false;
        }
        
        //this.ddmLevel = this.driftDetectionMethod.computeNextVal(prediction);
        this.driftDetectionMethod.input(prediction ? 0.0 : 1.0);
        this.ddmLevel = DDM_INCONTROL_LEVEL;
        if (this.driftDetectionMethod.getChange()) {
         this.ddmLevel =  DDM_OUTCONTROL_LEVEL;
        }
        if (this.driftDetectionMethod.getWarningZone()) {
           this.ddmLevel =  DDM_WARNING_LEVEL;
        }
        switch (this.ddmLevel) {
            case DDM_WARNING_LEVEL:
                //System.out.println("1 0 W");
            	//System.out.println("DDM_WARNING_LEVEL");
                if (newClassifierReset == true) {
                    this.warningDetected++;
                    this.newclassifier.resetLearning();
                    newClassifierReset = false;
                }
                this.newclassifier.trainOnInstance(inst);
                break;

            case DDM_OUTCONTROL_LEVEL:
                //System.out.println("0 1 O");
            	//System.out.println("DDM_OUTCONTROL_LEVEL");
                this.changeDetected++;
                this.classifier = null;
                this.classifier = this.newclassifier;
                if (this.classifier instanceof WEKAClassifier) {
                    ((WEKAClassifier) this.classifier).buildClassifier();
                }
                this.newclassifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
                this.newclassifier.resetLearning();
                getNewBoundary();
                eval.reset();
                break;

            case DDM_INCONTROL_LEVEL:
                //System.out.println("0 0 I");
            	//System.out.println("DDM_INCONTROL_LEVEL");
                newClassifierReset = true;
                break;
            default:
            //System.out.println("ERROR!");

        }

        this.classifier.trainOnInstance(inst);
    }
    
    @Override
    public double[] getVotesForInstance(Example<Instance> example){
    	double[] votes = this.classifier.getVotesForInstance(example);
		if(votes.length > 1)
			votes[1] = (votes[1]/decisionBoundary) - votes[1];
		return votes;
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
	
	public double getGMean(double positiveAccuracy, double negativeAccuracy) {
		return FastMath.sqrt(positiveAccuracy * negativeAccuracy);
	}
}
