import com.google.inject.internal.asm.$ByteVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import scala.Array;
import scala.Tuple2;






public class G10HW3
{

    static long SEED = 2020;

    public static  void main(String [] args)
    {
        // Throw exception for arguments which aren't <dataset, k, L>
        if (args.length != 3)
        {
            throw new IllegalArgumentException("USAGE: file_path k L");
        }

        // read arguments from line of command
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int L = Integer.parseInt(args[2]);

        // Create the Spark context
        SparkConf config = new SparkConf(true).setAppName("Homework3");

        // Instantiate the Spark context
        JavaSparkContext sc = new JavaSparkContext(config);
        sc.setLogLevel("WARN");

        long start_time = System.nanoTime(); // Start measuring time

        // Load every point (expressed as coordinates) as a Vector, then repartition it in L random subsets
        // Using partitions allows for enhanced parallelization
        JavaRDD<Vector> pointset = sc
                .textFile(filename)
                .map(p -> strToVector(p))
                .repartition(L)
                .cache();

        // Action to force loading
        pointset.count();

        long end_time = System.nanoTime(); // Start measuring time

        // Debug output
        long N = pointset.count();
        System.out.println("\nNumber of points = " + N);
        System.out.println("k =  " + k);
        System.out.println("L =   " + L);
        System.out.println("Initialization time = " + (end_time - start_time)/1000000 + " milliseconds");

        // Solve diversity maximization problem
        ArrayList<Vector> solution = runMapReduce(pointset, k, L);

        // Compute average distance
        double avg_dist = measure(solution);
        System.out.println("\nAverage distance = " + avg_dist);

    }


    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L)
    {

        long start_time = System.nanoTime(); // start measuring time

        ////// ROUND 1 //////
        JavaRDD<Vector> partitions = pointsRDD
                .mapPartitions( set ->
                {
                    ArrayList<Vector> points = new ArrayList<>();

                    // Push the points in the same partition in an arraylist
                    while(set.hasNext()) {
                        points.add(set.next());
                    }

                    // Compute farthest-traversal algorithm
                    ArrayList<Vector> k_points = kCenterMPD(points, k);

                    return k_points.iterator();

                });

        // Action to force loading
        partitions.count();

        long end_time = System.nanoTime(); // stop measuring time
        System.out.println("\nRuntime of Round 1 = " + (end_time - start_time)/1000000 + " milliseconds");

        ////// ROUND 2 //////
        start_time = System.nanoTime(); // start measuring time

        // Retrieve  set with all L*k points selected before
        ArrayList<Vector> coreset = new ArrayList<>(partitions.collect());

        // Run 2-approximate sequential algorithm
        ArrayList<Vector> seqPoints = runSequential(coreset, k);

        end_time = System.nanoTime(); // stop measuring time
        System.out.println("Runtime of Round 2 = " + (end_time - start_time)/1000000 + " milliseconds");

        return seqPoints;
    }

    // Compute the average distance between all pairs of points
    public static double measure(ArrayList<Vector> pointsSet)
    {

        int counter = 0;
        double distance = 0;

        // for each pair compute distance and add to the total
        for(int i = 0; i<pointsSet.size(); i++)
        {
            // the j=i+1 initialization prevents unnecessary calculations
            for(int j = i+1; j<pointsSet.size(); j++)
            {
                // calculate the euclidean distance of two point and add to the total distance
                distance += Math.sqrt(Vectors.sqdist(pointsSet.get(i), pointsSet.get(j)));
                counter+=1;
            }

        }

        return (distance/counter);

    }


    // RunSequential algorithm
    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        // Exit it k is larger than total size of P
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);

        for (int iter=0; iter<k/2; iter++)
        {

            double maxDist = 0;
            int maxI = 0, maxJ = 0;

            // Find the maximum distance pair among the candidates
            for (int i = 0; i < n; i++)
            {
                if (candidates[i])
                {
                    for (int j = i+1; j < n; j++)
                    {
                        if (candidates[j])
                        {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist)
                            {
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
        if (k % 2 != 0)
        {
            for (int i = 0; i < n; i++)
            {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }

        if (result.size() != k)
        {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;
    }

    // Process the CSV
    public static Vector strToVector(String str)
    {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // By using k-Center algorithm, given a pointset S and a number of cluster k
    // return an ArrayList of vectors containing the centers. Since we're only interested on the
    // centers and not the cluster, we won't track the nearest cluster for every point.
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
