#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

#define BUCKET_SIZE 10  // Maximum capacity of the bucket
#define OUT_RATE 3      // Number of packets sent per second (leak rate)

void leaky_bucket(int incoming_packets) {
    static int current_bucket_content = 0; // Packets currently in the bucket

    printf("\nIncoming Packets: %d\n", incoming_packets);

    // 1. Check for overflow
    if (incoming_packets + current_bucket_content > BUCKET_SIZE) {
        int dropped = (incoming_packets + current_bucket_content) - BUCKET_SIZE;
        printf("[OVERFLOW] Bucket full! Dropped %d packets.\n", dropped);
        current_bucket_content = BUCKET_SIZE; 
    } else {
        current_bucket_content += incoming_packets;
    }

    printf("Current Bucket Status: %d/%d\n", current_bucket_content, BUCKET_SIZE);

    // 2. Leak/Transmit packets
    if (current_bucket_content > 0) {
        int sent = (current_bucket_content < OUT_RATE) ? current_bucket_content : OUT_RATE;
        current_bucket_content -= sent;
        printf("[LEAK] Transmitted %d packets. Remaining in bucket: %d\n", sent, current_bucket_content);
    } else {
        printf("[IDLE] No packets to transmit.\n");
    }
    
    if (current_bucket_content == 0 && incoming_packets == 0) exit(0); 
}

int main() {

    srand(time(NULL));
    int i, packets;

    printf("--- Leaky Bucket Simulation ---\n");
    printf("Bucket Size: %d | Output Rate: %d pkts/sec\n", BUCKET_SIZE, OUT_RATE);

    // Simulate 5 seconds of traffic
    for (i = 1; i <= 5; i++) {
        packets = rand() % 10; // Randomly generate 0-9 packets
        printf("\n--- Second %d ---", i);
        leaky_bucket(packets);
        sleep(1); // Wait 1 second for the next "clock cycle"
    }
    
    while (1) {
        leaky_bucket(0);
        sleep(1);
    }


    return 0;
}

/*
--- Leaky Bucket Simulation ---
Bucket Size: 10 | Output Rate: 3 pkts/sec

--- Second 1 ---
Incoming Packets: 5
Current Bucket Status: 5/10
[LEAK] Transmitted 3 packets. Remaining in bucket: 2

--- Second 2 ---
Incoming Packets: 7
Current Bucket Status: 9/10
[LEAK] Transmitted 3 packets. Remaining in bucket: 6

--- Second 3 ---
Incoming Packets: 3
Current Bucket Status: 9/10
[LEAK] Transmitted 3 packets. Remaining in bucket: 6

--- Second 4 ---
Incoming Packets: 7
[OVERFLOW] Bucket full! Dropped 3 packets.
Current Bucket Status: 10/10
[LEAK] Transmitted 3 packets. Remaining in bucket: 7

--- Second 5 ---
Incoming Packets: 0
Current Bucket Status: 7/10
[LEAK] Transmitted 3 packets. Remaining in bucket: 4

Incoming Packets: 0
Current Bucket Status: 4/10
[LEAK] Transmitted 3 packets. Remaining in bucket: 1

Incoming Packets: 0
Current Bucket Status: 1/10
[LEAK] Transmitted 1 packets. Remaining in bucket: 0
*/


