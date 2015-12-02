package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        double before;
        double after;
        double diff;
        FixedIterationTrainer fit;
        RandomizedHillClimbing rhc;
         //for-loop to run 100 times
        System.out.println("Randomized Hill Climber Running");
        System.out.println("--------------------------------------------");
        for(int i = 1; i < 101; i++) {
            rhc = new RandomizedHillClimbing(hcp);      
            fit = new FixedIterationTrainer(rhc, i * 2000);
            before = System.currentTimeMillis() / 1000.0;
            fit.train();
            after = System.currentTimeMillis() / 1000.0;
            diff = after - before;
            System.out.println(ef.value(rhc.getOptimal()) + " " + diff);
        }
        

        //for-loop to run 100 times
        System.out.println("Simulated Annealing Running");
        System.out.println("--------------------------------------------");
        SimulatedAnnealing sa;
        for(int i = 1; i < 101; i++) {
            sa = new SimulatedAnnealing(1E12, .95, hcp);
            fit = new FixedIterationTrainer(sa, i * 2000); //originally 200,000
            before = System.currentTimeMillis() / 1000.0;
            fit.train();
            after = System.currentTimeMillis() / 1000.0;
            diff = after - before;
            System.out.println(ef.value(sa.getOptimal()) + " " + diff);
        }
        

        //for-loop to run 100 times
        System.out.println("Genetic Algorithm Running");
        System.out.println("--------------------------------------------");
        StandardGeneticAlgorithm ga;
        for(int i = 1; i < 101; i++) {
            ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
            fit = new FixedIterationTrainer(ga, i*10); //originally 1,000
            before = System.currentTimeMillis() / 1000.0;
            fit.train();
            after = System.currentTimeMillis() / 1000.0;
            diff = after - before;                
            System.out.println(ef.value(ga.getOptimal()) + " " + diff);
        }
        
        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        //for-loop to run 100 times
        System.out.println("MIMIC Running");
        System.out.println("--------------------------------------------");
        MIMIC mimic;
        for(int i = 1; i < 101; i++) {
            mimic = new MIMIC(200, 100, pop);
            fit = new FixedIterationTrainer(mimic, i*10); //originally 1,000
            before = System.currentTimeMillis() / 1000.0;
            fit.train();
            after = System.currentTimeMillis() / 1000.0;
            diff = after - before;
            System.out.println(ef.value(mimic.getOptimal()) + " " + diff);
        }
        
    }
}
