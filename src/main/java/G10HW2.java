import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.ArrayList;

class G10HW2 {
    public static  void main(String [] args) {
        //reads the points from the file into an ArrayList<Vector> called "inputPoints".

        /*
        Runs exactMPD(inputPoints), measuring its running time (in ms), and prints the following lines:
        EXACT ALGORITHM
        Max distance =  max distance returned by the method
        Running time =  running time of the method.
         */

        /*
        Runs twoApproxMPD(inputPoints,k), measuring its running time (in ms), and prints the following lines:
        2-APPROXIMATION ALGORITHM
        k =  value of k.
        Max distance =  max distance returned by the method
        Running time =  running time of the method.
         */

        /*
        Runs kcenterMPD(inputPoints,k), saves the returned points in an ArrayList<Vector> (list of tuple for Python users) called "centers", and runs exactMPD(centers), measuring the combined running time (in ms) of the two methods. Then, it prints the following lines:
        k-CENTER-BASED ALGORITHM
        k =  value of k.
        Max distance =  max distance returned by exactMPD(centers)
        Running time =  combined running time of the two methods.
         */
    }
    Long exactMPD(ArrayList<Vector> S) {
        return 0L;
    }
    Long twoApproxMPD(ArrayList<Vector> S, int k) {
        return 0L;
    }
    Long kCenterMPD(ArrayList<Vector> S, int k) {
        return 0L;
    }
}