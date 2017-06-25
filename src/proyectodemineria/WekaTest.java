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
import weka.core.Instances;
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
        
        /*Remove rm = new Remove();
        rm.setAttributeIndices("1");  // remove 1st attribute
        // classifier
        J48 j48 = new J48();
        j48.setUnpruned(true);        // using an unpruned J48
        // meta-classifier
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(rm);
        fc.setClassifier(j48);
        // train and make predictions
        fc.buildClassifier(train);
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = fc.classifyInstance(test.instance(i));
            System.out.print("ID: " + test.instance(i).value(0));
            System.out.print(", real: " + test.classAttribute().value((int) test.instance(i).classValue()));
            System.out.println(", predecido: " + test.classAttribute().value((int) pred));
          }
        */
        
    }
    
    public String classifyJ48(String datafile, int porcentaje) throws IOException, Exception{
        Instances data;
        try (BufferedReader datum = readDatafile(datafile)) {
            data = new Instances(datum);
        }
        
        data.setClassIndex(data.numAttributes()-1);
       
        int trainSize = (int) Math.round(data.numInstances() * porcentaje / 100);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        Classifier cls = new J48();
        cls.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        String results = eval.toSummaryString("\nResults\n======\n", false);
        return results;
    }
    public String classifyNaiveBayes(String datafile, int porcentaje) throws IOException, Exception{
        Instances data;
        try (BufferedReader datum = readDatafile(datafile)) {
            data = new Instances(datum);
        }
        
        data.setClassIndex(data.numAttributes()-1);
       
        int trainSize = (int) Math.round(data.numInstances() * porcentaje / 100);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        Classifier cls = new NaiveBayes();
        cls.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        String results = eval.toSummaryString("\nResults\n======\n", false);
        return results;
    }
    
    
    
}
