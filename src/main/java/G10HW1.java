import com.sun.xml.bind.v2.TODO;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class G10HW1 {

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: number_partitions, <path to file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> pairStrings = sc.textFile(args[1]).repartition(K);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Long> count;



        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CLASS COUNT 1st version
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        count = pairStrings
                .flatMapToPair((line) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = line.split(" ");
                    HashMap<String, String> pairsMap = new HashMap<>();


                    pairsMap.put(String.valueOf(Integer.parseInt(tokens[0])%K), tokens[1]);

                    ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
                    for (Map.Entry<String, String> e : pairsMap.entrySet()) {
                        pairs.add(new Tuple2<String,String>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()// <-- REDUCE PHASE (R1)
                .flatMapToPair((modPair) -> {
                    HashMap<String, Long> counts = new HashMap<>();
                    for (String token : modPair._2()) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });

        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        System.out.print("Output pairs = ");
        for(Tuple2<String, Long> tuple : count.sortByKey().collect()) {
            System.out.print("("+tuple._1()+","+tuple._2()+")");
        }
        System.out.print("\n");



        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CLASS COUNT 2nd version
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        count = pairStrings
                .flatMapToPair((line) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = line.split(" ");
                    HashMap<String, String> pairsMap = new HashMap<>();

                    pairsMap.put(String.valueOf(Integer.parseInt(tokens[0])%K), tokens[1]);

                    ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
                    for (Map.Entry<String, String> e : pairsMap.entrySet()) {
                        pairs.add(new Tuple2<String,String>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .mapPartitionsToPair((it) ->{// <-- REDUCE PHASE (R1)
                    HashMap<String, Long> counts = new HashMap<>();
                    while (it.hasNext()){
                        Tuple2<String, String> tuple = it.next();
                        counts.put(tuple._2(), 1L + counts.getOrDefault(tuple._2(), 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()  // <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                        long sum = 0;
                        for (long c : it) {
                            sum += c;
                        }
                        return sum;
                });


        Tuple2<String, Long> mostFrequent = new Tuple2<>("Z", -1L);

        for(Tuple2<String, Long> tuple : count.sortByKey().collect()) {
            if(tuple._2()>mostFrequent._2) mostFrequent=tuple;
        }
        System.out.println("VERSION WITH SPARK PARTITIONS");
        System.out.println("Most frequent class = ("+mostFrequent._1()+","+mostFrequent._2()+")");

        //TODO Max partition
    }

}

