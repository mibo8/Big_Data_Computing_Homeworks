import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Random;



public class G10HW2 {

    // SEED for reproducibility of random generation
    static long SEED = 1220423;

    // Given an ArrayList of Vector, calculate the maximum distance between two points
    public static double exactMPD(ArrayList<Vector> s)
    {
        double max_dist = 0.0;

        for(int i = 0; i <  s.size(); i++)
        {
            // the j=i+1 initialization prevents unnecessary calculations
            for(int j = i+1; j< s.size(); j++)
            {
                // calculate the squared distance of two point
                double dist = Vectors.sqdist(s.get(i), s.get(j));

                if(dist > max_dist) max_dist = dist;
            }
        }

        //return the square root of the maximum distance
        return Math.sqrt(max_dist);
    }

    // return the maximum distance between points from S t points in a random subset S' with size k
    public static double twoApproxMPD(ArrayList<Vector> s, int k)
    {
        // If k is smaller than S return
        if(k >= s.size())
        {
            System.out.println("k must be smaller than the size of S!!");
            return 0;
        }

        Random rand = new Random(SEED); // Initialize randomizer with constant SEED for reproducibility
        double max_dist = 0.0;
        ArrayList<Vector> s_prime = new ArrayList<>();

        // Create a random subset S' by taking k random points from S
        for(int i = 0; i < k; i++)
        {
            int rand_ind = rand.nextInt(s.size());
            s_prime.add(s.get(rand_ind));
            s.remove(rand_ind);
        }

        // calculate the maximum distance between points from S' and S
        for(int i = 0; i <  s_prime.size(); i++)
        {
            for(int j = 0; j< s.size(); j++)
            {
                // calculate the squared distance of two point
                double dist = Vectors.sqdist(s_prime.get(i), s.get(j));

                if(dist > max_dist) max_dist = dist;
            }
        }

        return Math.sqrt(max_dist);
    }


    // By using k-Center algorithm, given a pointset S and a number of cluster k
    // return an ArrayList of vectors containing the centers. Since we're only interested on the
    // centers and not the cluster, we won't track the nearest cluster for every point. This can be easily
    // extended for more generalized cases by using a Tuple3 which stores minimum distance from nearest center,
    // the nearest center and the coordinates of the point.
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

    // Return an ArrayList of Vector from the points in the file at the location 'filename'
    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException
    {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }


    public static  void main(String [] args)
    {
        // Check number of cluster and path of dataset given in input by line of command
        if (args.length != 2)
        {
            throw new IllegalArgumentException("USAGE: file_path k_value");
        }

        // read arguments from line of command
        String filename = args[0];
        int k = Integer.parseInt((args[1]));

        ArrayList<Vector> inputPoints = new ArrayList<>();

        // Return an ArrayList of Vector from the points in the file at the location 'filename'
        // try catch block is needed for exception
        try {
            inputPoints = readVectorsSeq(filename);
        }
        catch(IOException e) {
            System.out.println (e.toString());
        }

        // Exact algorithm execution; Print the result and the execution time in milliseconds
        // To measure the execution time we consider the difference in time in nanoseconds
        // prior and after the call to the function and then divide it for 1000000
        long start_time = System.nanoTime();
        double max_distance_exact = exactMPD(inputPoints);
        long end_time = System.nanoTime();
        System.out.println("EXACT ALGORITHM");
        System.out.println("Max distance = " + max_distance_exact);
        System.out.println("Running time = " + (end_time - start_time)/1000000 + " milliseconds");

        // Approximated algorithm execution; Print the results and the execution time in milliseconds
        // To measure the execution time we consider the difference in time in nanoseconds
        // prior and after the call to the function and then divide it for 1000000
        start_time = System.nanoTime();
        double max_distance_approx = twoApproxMPD(inputPoints, k);
        end_time = System.nanoTime();
        System.out.println("2-APPROXIMATION ALGORITHM");
        System.out.println("k = "+k);
        System.out.println("Max distance = " + max_distance_approx);
        System.out.println("Running time = " + (end_time - start_time)/1000000 + " milliseconds");

        // K-Center algorithm execution; Print the results and the execution time in milliseconds
        // To measure the execution time we consider the difference in time in nanoseconds
        // prior and after the call to the function and then divide it for 1000000
        start_time = System.nanoTime();
        ArrayList<Vector> centers = kCenterMPD(inputPoints, k);
        double max_distance_fft = exactMPD(centers);
        end_time = System.nanoTime();
        System.out.println("k-CENTER-BASED ALGORITHM");
        System.out.println("k = "+k);
        System.out.println("Max distance = " + max_distance_fft);
        System.out.println("Running time = " + (end_time - start_time)/1000000 + " milliseconds");


        }


}
