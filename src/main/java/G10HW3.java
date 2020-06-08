import org.apache.hadoop.mapreduce.v2.app.speculate.TaskRuntimeEstimator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.storage.StorageLevel;
import org.dmg.pmml.True;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;


public class G10HW3 {

    public static  void main(String [] args) {

        // Check number of partition and path of dataset given in input by line of command
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path k num_partitions");
        }

        // Create the Spark context
        SparkConf config = new SparkConf(true).setAppName("Homework3");
        // Instantiate the Spark context
        JavaSparkContext sc = new JavaSparkContext(config);
        sc.setLogLevel("WARN");

        // Read k
        int k = Integer.parseInt(args[1]);

        // Read number of partitions
        int L = Integer.parseInt(args[2]);

        //Initialization and time measurement
        long startTime = System.nanoTime();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0]).map(str -> strToVector(str)).repartition(L).cache();

        // Action to force loading
        inputPoints.count();

        //Time measurement
        long endTime   = System.nanoTime();
        long totalTime = (endTime - startTime)/1000000;

        System.out.println("Number of points = " + inputPoints.count());
        System.out.println("k = " + k);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + totalTime +" ms");

        ArrayList<Vector> results = runMapReduce(inputPoints, k ,L);

        //Compute the average distance between output points
        System.out.println("Average distance: " + measure(results));

    }
    public  static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L){
        //time measurement
        long startTime = System.nanoTime();

        //-----ROUND 1-----
        JavaRDD<Vector> candidates = pointsRDD.mapPartitions(x -> {
            // Create a list for each partition
            ArrayList<Vector> pointsList = new ArrayList<>();
            while(x.hasNext()) pointsList.add(x.next());
            // Compute centers with Farthest-First Traversal
            ArrayList<Vector> centers = kCenterMPD(pointsList, k);
            // return an iterator to the centers list
            return centers.iterator();
        });

        // Action to force loading
        candidates.count();

        //time measurement
        long endTime   = System.nanoTime();
        long totalTime = (endTime - startTime)/1000000;
        System.out.println("Runtime of Round 1 = " + totalTime +" ms");

        //-----ROUND 2-----
        startTime = System.nanoTime();

        //Collect candidates into a single list
        ArrayList<Vector> coreset = new ArrayList<Vector>(candidates.collect());
        //Compute Diversity Maximization woth the subset in input
        ArrayList<Vector> results = runSequential(coreset, k);

        // Time measurement
        endTime   = System.nanoTime();
        totalTime = (endTime - startTime)/1000000;
        System.out.println("Runtime of Round 2 = " + totalTime +" ms");

        // Output
        return results;
    }

    //Compute the average distance
    public static double measure(ArrayList<Vector> pointsSet){
        double sum = 0;
        int n = pointsSet.size();
        for(int i=0; i<n ;i++) {
            for(int j=i+1; j<n; j++) {
                sum += Math.sqrt(Vectors.sqdist(pointsSet.get(i),pointsSet.get(j)));
            }
        }
        return sum/((n*(n-1))/2);

    }


    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    }
    // END runSequential

    // Auxiliary methods

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> s, int k)
    {

        ArrayList<Vector> centers = new ArrayList<>();
        ArrayList<Double> distances = new ArrayList<>();

        // Add an arbitrary first center (index 0 for example)
        centers.add(s.get(0));
        // remove the center from S
        s.remove(0);

        // For every centroid, track the minimum distances between every point in S and every center
        // Then we can add the point with the maximum distance from the nearest center
        for(int i = 0; i < k-1; i++)
        {

            double max_dist = 0.0;
            int new_cluster = 0; // flag for tracking what center add next

            for(int j = 0; j < s.size(); j++)
            {
                // Calculate the distance between the point and the last centroid added
                double dist = Vectors.sqdist(s.get(j), centers.get(i));

                // Only first cycle: fill the ArrayList with the distances from the arbitrary center
                if (i == 0)
                {
                    distances.add(dist);

                    // find the second cluster
                    if (max_dist < distances.get(j))
                    {
                        max_dist = distances.get(j);
                        new_cluster = j;
                    }
                }

                // Otherwise
                else {
                    // If the distance between the new centroid and the point is less than the previous minimum distance,
                    // update the minimum distance and check for a possible centroid candidate
                    if(dist < distances.get(j))
                    {
                        distances.set(j, dist);
                        if (max_dist < distances.get(j)) {
                            max_dist = distances.get(j);
                            new_cluster = j;
                        }

                    }
                    // Otherwise, check if the previous distance has the maximum distance
                    else if (max_dist < distances.get(j))
                    {
                        max_dist = distances.get(j);
                        new_cluster = j;
                    }

                }

            }

            // Add the new centroid to the Arraylist
            centers.add(s.get(new_cluster));
            // Remove the centroid from the set S of remaining points and the distance tracker
            s.remove(new_cluster);
            distances.remove(new_cluster);

        }
        return centers;
    }

}