import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

class G10HW22 {
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

            /*long startTime = System.nanoTime();
            double exactDistance = exactMPD(inputPoints);
            long endTime   = System.nanoTime();
            long totalTime = (endTime - startTime)/1000000;

            System.out.println("EXACT ALGORITHM");
            System.out.println("Max distance = " + exactDistance);
            System.out.println("Running time = " + totalTime + " ms");*/


            /*
            Runs twoApproxMPD(inputPoints,k), measuring its running time (in ms), and prints the following lines:
            2-APPROXIMATION ALGORITHM
            k =  value of k.
            Max distance =  max distance returned by the method
            Running time =  running time of the method.
             */


            int k = Integer.parseInt(args[1]);

            long startTime = System.nanoTime();
            double approxDistance = twoApproxMPD(inputPoints, k);
            long endTime   = System.nanoTime();
            long totalTime = (endTime - startTime)/1000000;

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
        randomGenerator.setSeed(1220423);
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


    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> S, int k) {
        ArrayList<Vector> C = new ArrayList<>();
        ArrayList<Double> distances = new ArrayList<>();
        C.add(S.get(0));
        S.remove(0);
        for(int i=0; i<S.size(); i++) {
            distances.add(Math.sqrt(Vectors.sqdist(S.get(i), C.get(0))));
        }
        for(int i=1; i<k; i++) {
            double maxDistance = -1;
            int centerIndex = 0;
            for (int j=0; j<S.size(); j++) {
                double d = Math.sqrt(Vectors.sqdist(S.get(j), S.get(centerIndex)));
                if(d < distances.get(j)){
                    distances.set(j, d);
                }
                if(distances.get(j) > maxDistance) {
                    maxDistance = distances.get(j);
                    centerIndex = j;
                }
            }
            C.add(S.get(centerIndex));
            S.remove(centerIndex);
            distances.remove(centerIndex);
        }
        return C;
    }
}