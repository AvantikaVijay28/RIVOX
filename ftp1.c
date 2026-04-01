ser
#include<stdio.h> // For printf(), fopen()
#include<sys/types.h> // Data types used in socket programming
#include<sys/socket.h> // For socket(), bind(), listen(), accept()
#include<netinet/in.h> // For sockaddr_in structure
#include<string.h> // For strcmp(), strlen(), memset()
#include<unistd.h> // For read(), write(), close()
#include<stdlib.h> // For exit()

char buff[4096]; // Buffer to store file data temporarily
FILE *f1; // File pointer to read file
int MAX=60; // Maximum size for filename communication

int main(){

    int sock_fd,newSock_fd,clength;

    // serv_addr → server address
    // cli_addr → client address
    struct sockaddr_in serv_addr,cli_addr;

    char str[60]; // To store filename sent by client

    // Create TCP socket
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    // AF_INET → IPv4
    // SOCK_STREAM → TCP protocol

    if(sock_fd<0)
        printf("Cannot create socket\n");
    else
        printf("Socket created\n");        

    // Setting server address properties
    serv_addr.sin_family = AF_INET; // IPv4
    serv_addr.sin_addr.s_addr = INADDR_ANY; // Accept from any IP
    serv_addr.sin_port = htons(3000); // Port number 3000

    // Bind socket to IP and port
    bind(sock_fd,(struct sockaddr*)&serv_addr,sizeof(serv_addr));
    printf("Binded\n");

    // Server waits for client connection
    listen(sock_fd,10);

    clength=sizeof(cli_addr);

    // Accept connection from client
    newSock_fd=accept(sock_fd,(struct sockaddr*)&cli_addr,(socklen_t*)&clength);

    close(sock_fd); // No need original socket after accept

    do{
        memset(str,0,sizeof(str)); // Clear previous data

        // Read filename sent by client
        read(newSock_fd,str,60);

        // If client sends exit → close connection
        if(strcmp(str,"exit")==0){
            close(newSock_fd);
            exit(0);
        }

        printf("\nFile Requested: %s\n",str);

        // Try opening requested file
        f1=fopen(str,"r");

        if(f1==NULL){
            // If file not found
            char err[]="ERROR OCCURED! NO FILE FOUND";
            write(newSock_fd,err,strlen(err));
        }
        else{
            // Read file line by line
            while(fgets(buff,4096,f1)!=NULL){

                // Send file content to client
                write(newSock_fd,buff,strlen(buff));
            }
            fclose(f1); // Close file after sending
        }

    }while(1);

    return 0;
}
