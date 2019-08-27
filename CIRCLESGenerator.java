package moa.streams.generators;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;

import moa.core.Example;
import moa.core.FastVector;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.util.Random;

import moa.core.InstanceExample;

import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

public class CIRCLESGenerator extends AbstractOptionHandler implements InstanceStream {

    protected InstancesHeader streamHeader;
    protected int instancesGenerated = 0;
	private int startCircle;
	private int endCircle;
	protected int driftPoint;
	protected double driftSlope;
	private long randomNumberSeed;
	protected boolean invert = false;
	
	protected Random rng;
	
	protected double oldRadius;
	protected double newRadius;
    
	@Override
	public String getPurposeString() {
		return "Generates CIRCLES concept functions.";
	}
	
	private static final long serialVersionUID = 1L;
	
	// Circles: centre = {0.5, 0.5}, radius: 0 -> 0.2, 1 -> 0.25; 2 -> 0.3; 3 -> 0.35; 4 -> 0.4
    public IntOption startCircleOption = new IntOption("startCircle", 's',
            "Starting function for generating instances",
            0, 0, 4);
	
    public IntOption endCircleOption = new IntOption("endCircle", 'e',
            "End function for generating instances",
            1, 0, 4);
    
    public IntOption driftPointOption = new IntOption("driftPoint", 'd',
            "Num of instances before starting drift",
            999000);
    
    public FlagOption invertOption = new FlagOption("invert",
            'v', "Reverse ordering of classes.");
    
    // Increase in probability of using new concept by time-step once driftPoint is reached
    public FloatOption driftSlopeOption = new FloatOption("driftSlope", 'r',
            "Speed in introducing new concept",
            0.001, 0, 1);
    
    // RandomNumberSeed
    public IntOption instanceRandomSeedOption = new IntOption("instanceRandomSeed", 'i',
            "Seed for generating random numbers",
            1);

	@Override
	public InstancesHeader getHeader() {
        return this.streamHeader;
	}

	public void setStartCircle(Integer i){
		this.startCircle = i;
		this.oldRadius = startCircle * 0.05 + 0.2;
	}
	public void setEndCircle(Integer i){
		this.endCircle = i;
		this.newRadius = endCircle * 0.05 + 0.2;
	}
	public void setDriftSlope(Double d){
		this.driftSlope = d;
	}
	public void setDriftPoint(Integer i){
		this.driftPoint = i;
	}
	public void setRandomNumberSeed(Integer i){
		this.randomNumberSeed = i;
		this.rng = new Random(randomNumberSeed);
	}
	
	@Override
	public long estimatedRemainingInstances() {
		return -1;
	}

	@Override
	public boolean hasMoreInstances() {
		return true;
	}

	@Override
	public InstanceExample nextInstance() {
		double x = 0;
		double y = 0;
		double group = 0, concept = 0;
		
		// decide class and concept
		group = rng.nextDouble() <= 0.5 ? 1 : 0;
		double newConceptProb = Math.max(0, (instancesGenerated - driftPoint) * driftSlope);
		double radius = rng.nextDouble() <= newConceptProb ? newRadius : oldRadius;
		
		if(radius == newRadius){
			//System.out.println("gogo");
		}
		
		// find coordinates
		// uses http://stackoverflow.com/questions/481144/equation-for-testing-if-a-point-is-inside-a-circle
		boolean validValues = false;
		while(!validValues){
			x = rng.nextDouble();
			y = rng.nextDouble();
			if(Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2) < Math.pow(radius, 2) & group == 1){
				validValues = true;
			} else if (Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2) > Math.pow(radius, 2) & group == 0){
				validValues = true;
			}			
		}
		
        // construct instance
        InstancesHeader header = getHeader();
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setValue(0, x);
        inst.setValue(1, y);
        inst.setDataset(header);
        if(invert) group = 1 - group;
        inst.setClassValue(group);
        
        instancesGenerated++;
        
        return new InstanceExample(inst);
	}

	@Override
	public boolean isRestartable() {
		return true;
	}

	@Override
	public void restart() {
		instancesGenerated = 0;
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {
        
		// reset stream generating characteristics
		setStartCircle(startCircleOption.getValue());
		setEndCircle(endCircleOption.getValue());
		setDriftPoint(driftPointOption.getValue());
		setDriftSlope(driftSlopeOption.getValue());
		setRandomNumberSeed(instanceRandomSeedOption.getValue());
		
		
		// generate header
		FastVector attributes = new FastVector();
        attributes.addElement(new Attribute("x"));
        attributes.addElement(new Attribute("y"));

        FastVector classLabels = new FastVector();
        for (int i = 0; i < 2; i++) {
            classLabels.addElement("class" + (i + 1));
        }
        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
		if (this.invertOption.isSet()) {
	           invert = true;
	        }
        restart();
	}
}


