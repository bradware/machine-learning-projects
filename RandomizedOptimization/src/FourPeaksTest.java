package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 1000;
    /** The t value */
    private static final int T = N / 5;

    public static void main(String[] args) {
        
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        

        double before;
        double after;
        double diff;
        FixedIterationTrainer fit;

        RandomizedHillClimbing rhc;
        for(int i = 1; i < 101; i++) {
            rhc = new RandomizedHillClimbing(hcp);      
            fit = new FixedIterationTrainer(rhc, i * 2000); //originally 200,000
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
            sa = new SimulatedAnnealing(1E11, .95, hcp);
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
            ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
            fit = new FixedIterationTrainer(ga, i*10);    //originally 1,000
            before = System.currentTimeMillis() / 1000.0;
            fit.train();
            after = System.currentTimeMillis() / 1000.0;
            diff = after - before;
            System.out.println(ef.value(ga.getOptimal()) + " " + diff);
        }
        

        //for-loop to run 100 times
        System.out.println("MIMIC Running");
        System.out.println("--------------------------------------------");
        MIMIC mimic;
        for(int i = 1; i < 101; i++) {
            mimic = new MIMIC(200, 20, pop);
            fit = new FixedIterationTrainer(mimic, i * 10); //originally 1,000
            before = System.currentTimeMillis() / 1000.0;
            fit.train();
            after = System.currentTimeMillis() / 1000.0;
            diff = after - before;
            System.out.println(ef.value(mimic.getOptimal()) + " " + diff);
        }
    }
}
