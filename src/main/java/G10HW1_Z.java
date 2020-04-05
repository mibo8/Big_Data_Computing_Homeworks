import org.apache.hadoop.mapreduce.v2.app.speculate.TaskRuntimeEstimator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.dmg.pmml.True;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;


public class G10HW1_Z {

    public static  void main(String [] args) {

        // Check number of partition and path of dataset given in input by line of command
        if (args.length != 2)
        {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // Create the Spark context
        SparkConf config = new SparkConf(true).setAppName("Homework1");
        // Instantiate the Spark context
        JavaSparkContext sc = new JavaSparkContext(config);
        sc.setLogLevel("WARN");

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> pairStrings = sc.textFile(args[1]).repartition(K);

        JavaPairRDD<String, Long> classcount;

        classcount = pairStrings
                // ROUND 1: MAP PHASE
                .mapToPair((entry) ->
                {
                    // Split the input strings in the pair <key, class>
                    String[] tokens = entry.split(" ");
                    // return the new key-value pair
                    return new Tuple2<Integer, String>(Integer.parseInt(tokens[0])%K, tokens[1]);
                })

                //ROUND 1: REDUCE PHASE
                // Group every entry with the same key in a JavaPairRDD<K,Iterable<V>>
                .groupByKey()
                .flatMapToPair((entries) ->
                {
                    // Create an intermediate Hashmap which stores the occurrences
                    HashMap<String, Integer> counter = new HashMap<>();
                    // fill the Hash Map
                    for(String entry: entries._2())
                    {
                        counter.put(entry, 1+counter.getOrDefault(entry, 0));
                    }
                    // move the hasmap into a suitable type ArrayList
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    for(Map.Entry<String, Integer> hm : counter.entrySet())
                    {
                       pairs.add(new Tuple2<>(hm.getKey(), hm.getValue()));
                    }
                    return pairs.iterator();

                })
                //ROUND 2: REDUCE PHASE
                // Group every entry with the same key in a JavaPairRDD<K,Iterable<V>>
                .groupByKey()
                // Calculate the sum of the occurances of every entry
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });

        // Print deterministic partition output
        System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
        System.out.print("Output pairs = ");
        for(Tuple2<String, Long> tuple : classcount.sortByKey().collect()) {
            System.out.print("("+tuple._1()+", "+tuple._2()+") ");
        }
        System.out.print("\n");


        classcount = pairStrings
                //ROUND 1: MAP PHASE
                .mapToPair((entry) ->
                {
                    // Split the input strings in the pair <key, class>
                    String[] tokens = entry.split(" ");
                    // return the new key-value pair
                    return new Tuple2<Integer, String>(Integer.parseInt(tokens[0])%K, tokens[1]);
                })

                //ROUND 1: REDUCE PHASE
                // Runs map transformations on every partition of the RDD
                .mapPartitionsToPair((cc) -> {

                    HashMap<String, Integer> counter = new HashMap<>();
                    int nmax = 0; //nmax counter

                    // Count the occurrences of the same object in the partition
                    while(cc.hasNext())
                    {
                        nmax+=1;
                        Tuple2<Integer, String> tuple = cc.next();
                        counter.put(tuple._2, 1+counter.getOrDefault(tuple._2, 0));
                    }
                    // move the hasmap into a suitable type ArrayList
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    for(Map.Entry<String, Integer> hm : counter.entrySet())
                    {
                        pairs.add(new Tuple2<>(hm.getKey(), hm.getValue()));
                    }
                    // add the <"maxPartitionSize", nmax> pair
                    pairs.add(new Tuple2<>("maxPartitionSize", nmax));

                    return pairs.iterator();

                })
                 // Group every entry with the same key in a JavaPairRDD<K,Iterable<V>>
                .groupByKey()
                // If the entry is an object of our dataset, sums up the occurrences,
                // Otherwise we select the larger nmax in the subset with key = "maxPartitionsize"
                .mapToPair((entry) -> {
                    if(entry._1().equals("maxPartitionSize") == false)
                    {
                        long sum = 0;
                        for (long c : entry._2()) {
                            sum += c;
                        }
                        return new Tuple2<String, Long>(entry._1(), sum);
                    } else
                        {
                        long nmax = 0;
                        for(long c : entry._2()) {
                            if (c > nmax) nmax = c;
                        }
                        return new Tuple2<String, Long>(entry._1(), nmax);
                    }
                });

        // Check the most frequent class and the nmax of data processed in a partition
        Tuple2<String, Long> mostFrequent = new Tuple2<>("Z", -1L);
        long maxPartition = 0;

        for(Tuple2<String, Long> tuple : classcount.sortByKey().collect()) {
            if(tuple._1.equals("maxPartitionSize")) {
                maxPartition = tuple._2;
            } else if(tuple._2()>mostFrequent._2) mostFrequent=tuple;
        }
        // Output the spark partition results
        System.out.println("VERSION WITH SPARK PARTITIONS");
        System.out.println("Most frequent class = ("+mostFrequent._1()+","+mostFrequent._2()+")");
        System.out.println("Max partition size = "+maxPartition);

    }

}
