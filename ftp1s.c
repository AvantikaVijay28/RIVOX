Ftp client 
#include<stdio.h>        
#include<sys/types.h>    
#include<sys/socket.h>   
#include<netinet/in.h>   
#include<string.h>       
#include<unistd.h>       
#include<stdlib.h>
#include<arpa/inet.h> // For inet_addr()

int MAX=60;

int main(){

    int sock_fd,n;
    struct sockaddr_in serv_addr;

    char send[MAX],recvline[MAX];

    // Create TCP socket
    sock_fd=socket(AF_INET,SOCK_STREAM,0);

    if(sock_fd<0)
        printf("Cannot create socket\n");
    else
        printf("Socket created\n");

    // Set server details
    serv_addr.sin_family=AF_INET;
    serv_addr.sin_port=htons(3000);

    // Server IP address (localhost)
    serv_addr.sin_addr.s_addr=inet_addr("127.0.0.1");

    // Connect to server
    connect(sock_fd,(struct sockaddr*)&serv_addr,sizeof(serv_addr));

    while(1){

        // Ask user for file name
        printf("\nEnter the source file name : ");
        scanf("%s",send);

        // Send filename to server
        write(sock_fd,send,strlen(send));

        // If user types exit → close client
        if(strcmp(send,"exit")==0){
            exit(0);
        }

        memset(recvline,0,sizeof(recvline));

        // Receive file content from server
        n=read(sock_fd,recvline,MAX);

        recvline[n]='\0'; // Proper string termination

        // Print received data
        printf("\n%s\n",recvline);
    }

    close(sock_fd);
    return 0;
}

