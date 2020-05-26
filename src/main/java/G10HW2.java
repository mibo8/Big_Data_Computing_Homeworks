import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

class G10HW2 {
    public static void main(String [] args) {

        // Reading points from a file whose name is provided as args[0]

        String filename = args[0];
        try {
            ArrayList<Vector> inputPoints = new ArrayList<>(readVectorsSeq(filename));


            /*
            Runs exactMPD(inputPoints), measuring its running time (in ms), and prints the following lines:
            EXACT ALGORITHM
            Max distance =  max distance returned by the method
            Running time =  running time of the method.
             */

            long startTime = System.nanoTime();
            double exactDistance = exactMPD(inputPoints);
            long endTime   = System.nanoTime();
            long totalTime = (endTime - startTime)/1000000;

            System.out.println("EXACT ALGORITHM");
            System.out.println("Max distance = " + exactDistance);
            System.out.println("Running time = " + totalTime + " ms");


            /*
            Runs twoApproxMPD(inputPoints,k), measuring its running time (in ms), and prints the following lines:
            2-APPROXIMATION ALGORITHM
            k =  value of k.
            Max distance =  max distance returned by the method
            Running time =  running time of the method.
             */


            int k = Integer.parseInt(args[1]);

            startTime = System.nanoTime();
            double approxDistance = twoApproxMPD(inputPoints, k);
            endTime   = System.nanoTime();
            totalTime = (endTime - startTime)/1000000;

            System.out.println("\n2-APPROXIMATION ALGORITHM");
            System.out.println("k = " + k);
            System.out.println("Max distance = " + approxDistance);
            System.out.println("Running time = " + totalTime + " ms");

            /*
            Runs kcenterMPD(inputPoints,k), saves the returned points in an ArrayList<Vector> (list of tuple for Python users) called "centers", and runs exactMPD(centers), measuring the combined running time (in ms) of the two methods. Then, it prints the following lines:
            k-CENTER-BASED ALGORITHM
            k =  value of k.
            Max distance =  max distance returned by exactMPD(centers)
            Running time =  combined running time of the two methods.
             */

            inputPoints = new ArrayList<>(readVectorsSeq(filename));

            startTime = System.nanoTime();
            ArrayList<Vector> C = kCenterMPD(inputPoints, k);
            double exactCentersDistance = exactMPD(C);
            endTime   = System.nanoTime();
            totalTime = (endTime - startTime)/1000000;

            System.out.println("\nk-CENTER-BASED ALGORITHM");
            System.out.println("k = " + k);
            System.out.println("Max distance = " + exactCentersDistance);
            System.out.println("Running time = " + totalTime + " ms");

        }
        catch(Exception e) {
            e.printStackTrace();
        }

    }

    // Auxiliary methods

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

    //MPD methods

    public static double exactMPD(ArrayList<Vector> S) {
        double maxDistance = -1;
        for(Vector x : S) {
            for(Vector y : S) {
                double d = Math.sqrt(Vectors.sqdist(x,y));
                if (d > maxDistance) maxDistance=d;
            }
        }
        return maxDistance;
    }

    public static double twoApproxMPD(ArrayList<Vector> S, int k) {
        double maxDistance = -1;
        Random randomGenerator = new Random();
        randomGenerator.setSeed(1242457L);
        ArrayList<Vector> Sprime = new ArrayList<Vector>();
        for (int i=0; i<k; i++) {
            int randomIndex = randomGenerator.nextInt(S.size());
            Sprime.add(S.get(randomIndex));
            S.remove(randomIndex);
        }
        for (Vector x : Sprime) {
            for(Vector y : S) {
                double d = Math.sqrt(Vectors.sqdist(x,y));
                if (d > maxDistance) maxDistance=d;
            }
        }
        return maxDistance;
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