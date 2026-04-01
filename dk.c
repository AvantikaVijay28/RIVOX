#include <stdio.h>

struct node {
    int dist[26]; // Increased to 26 for letters A-Z
    int from[26];
} route[26];

int main() {
    int dm[26][26];
    int n, i, j, k, count = 0;

    printf("Enter the number of nodes (up to 26): ");
    scanf("%d", &n);

    printf("Enter the cost matrix (use 999 for infinity):\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            scanf("%d", &dm[i][j]);
            dm[i][i] = 0;
            route[i].dist[j] = dm[i][j];
            route[i].from[j] = j; // Initial next hop is the destination itself
        }
    }

    

    // Distance Vector Update Logic
    do {
        count = 0;
        for (i = 0; i < n; i++) {           // Source Node
            for (j = 0; j < n; j++) {       // Destination Node
                for (k = 0; k < n; k++) {   // Neighbor Node
                    if (route[i].dist[j] > dm[i][k] + route[k].dist[j]) {
                        route[i].dist[j] = dm[i][k] + route[k].dist[j];
                        route[i].from[j] = k;
                        count++;
                    }
                }
            }
        }
    } while (count != 0);

    // Displaying the final routing tables using characters
    for (i = 0; i < n; i++) {
        printf("\n\nRouting table for Node %c:\n", i + 'A');
        printf("Dest\tNext Hop\tDist\n");
        for (j = 0; j < n; j++) {
            // Convert index j and from[j] back to characters (0 -> A, 1 -> B)
            printf("%c\t%c\t\t%d\n", j + 'A', route[i].from[j] + 'A', route[i].dist[j]);
        }
    }

    return 0;
}

/*
Enter the number of nodes (up to 26): 4
Enter the cost matrix (use 999 for infinity):
0 2 999 1
2 0 2 7
999 2 0 11
1 7 11 0


Routing table for Node A:
Dest	Next Hop	Dist
A	A		0
B	B		2
C	B		4
D	D		1


Routing table for Node B:
Dest	Next Hop	Dist
A	A		2
B	B		0
C	C		2
D	A		3


Routing table for Node C:
Dest	Next Hop	Dist
A	B		4
B	B		2
C	C		0
D	B		5


Routing table for Node D:
Dest	Next Hop	Dist
A	A		1
B	A		3
C	A		5
D	D		0
/*


