/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package proyectodemineria;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import static weka.core.Instances.test;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author anselcorona
 */
public class WekaTest {
    public static BufferedReader readDatafile(String filename){
        BufferedReader inputreader = null;
        
        try{
            inputreader = new BufferedReader(new FileReader(filename));
        }catch(Exception ex){
            ex.printStackTrace();
        }
        return inputreader;
    }
    
    public static void main(String[] args) throws Exception{
        BufferedReader datafile = readDatafile("iris.arff");
        Instances data = new Instances(datafile);
        datafile.close();
        
        data.setClassIndex(data.numAttributes()-1);
       
        int trainSize = (int) Math.round(data.numInstances() * 77 / 100);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        Classifier cls = new J48();
        cls.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        
        
        
    }
    
    public String classifyJ48(String datafile) throws IOException, Exception{
        DataSource source = new DataSource("training/" + datafile);
        Instances trainingset = source.getDataSet();
        trainingset.setClassIndex(trainingset.numAttributes()-1);
        Classifier cls = new J48();
        cls.buildClassifier(trainingset);
        
        DataSource source1 = new DataSource("testing/" + datafile);
        Instances testDataset = source1.getDataSet();
        testDataset.setClassIndex(testDataset.numAttributes()-1);
        Evaluation eval = new Evaluation(trainingset);
        eval.evaluateModel(cls, testDataset);
        
        String results="";
        results+="===================\n";
        results+="Actual Class, J48 Predicted\n";
        for (int i = 0; i < testDataset.numInstances(); i++) {
                //get class double value for current instance
                double actualValue = testDataset.instance(i).classValue();

                //get Instance object of current instance
                Instance newInst = testDataset.instance(i);
                //call classifyInstance, which returns a double value for the class
                double predJ48 = cls.classifyInstance(newInst);

                results += "Real: " + testDataset.classAttribute().value((int) testDataset.instance(i).classValue()) + ". Predicted:" +  testDataset.classAttribute().value((int) predJ48) + "\n";
        }
        
        results += eval.toSummaryString("\nResults\n======\n", false);
        
        
        return results;
    }
    public String classifyNaiveBayes(String datafile) throws IOException, Exception{
        DataSource source = new DataSource("training/" + datafile);
        Instances trainingset = source.getDataSet();
        trainingset.setClassIndex(trainingset.numAttributes()-1);
        Classifier cls = new NaiveBayes();
        cls.buildClassifier(trainingset);
        
        DataSource source1 = new DataSource("testing/" + datafile);
        Instances testDataset = source1.getDataSet();
        testDataset.setClassIndex(testDataset.numAttributes()-1);
        Evaluation eval = new Evaluation(trainingset);
        eval.evaluateModel(cls, testDataset);
        
        String results="";
        results+="===================\n";
        results+="Actual Class    | Naive Bayes Predicted\n";
        for (int i = 0; i < testDataset.numInstances(); i++) {
                //get class double value for current instance
                double actualValue = testDataset.instance(i).classValue();

                //get Instance object of current instance
                Instance newInst = testDataset.instance(i);
                //call classifyInstance, which returns a double value for the class
                double predNB = cls.classifyInstance(newInst);

                results += "Real: " + testDataset.classAttribute().value((int) testDataset.instance(i).classValue()) + ". Predicted:" +  testDataset.classAttribute().value((int) predNB) + "\n";
        }
        
        results += eval.toSummaryString("\nResults\n======\n", false);
        
        
        return results;
    }
    public String classifyRandomForest(String datafile) throws IOException, Exception{
        DataSource source = new DataSource("training/" + datafile);
        Instances trainingset = source.getDataSet();
        trainingset.setClassIndex(trainingset.numAttributes()-1);
        Classifier cls = new RandomForest();
        cls.buildClassifier(trainingset);
        
        DataSource source1 = new DataSource("testing/" + datafile);
        Instances testDataset = source1.getDataSet();
        testDataset.setClassIndex(testDataset.numAttributes()-1);
        Evaluation eval = new Evaluation(trainingset);
        eval.evaluateModel(cls, testDataset);
        
        String results="";
        results+="===================\n";
        results+="Actual Class    | Random Forest Predicted\n";
        for (int i = 0; i < testDataset.numInstances(); i++) {
                //get class double value for current instance
                double actualValue = testDataset.instance(i).classValue();

                //get Instance object of current instance
                Instance newInst = testDataset.instance(i);
                //call classifyInstance, which returns a double value for the class
                double predRF = cls.classifyInstance(newInst);

                results += "Real: " + testDataset.classAttribute().value((int) testDataset.instance(i).classValue()) + ". Predicted:" +  testDataset.classAttribute().value((int) predRF) + "\n";
        }
        
        results += eval.toSummaryString("\nResults\n======\n", false);
        
        
        return results;
    }
    
    
    
}
