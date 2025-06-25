pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "mlops-new-447207"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
    }
    stages{
        /* Stage 1 */
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins............'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/data-guru0/MLOPS-COURSE-PROJECT-1.git']])
                }
            }
        }
        /* Stage 2 */
        stage('Setting up our Virtual Environment and Installing dependancies'){
            steps{
                script{
                    echo 'Setting up our Virtual Environment and Installing dependancies............'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
        /* Stage 3 */
        stage('Building and Pushing Docker Image to GCR'){
            steps{
                /* file(variable: ID) ID go to Jenkins dashboard > Manage Jenkins > Credentials > System > Global credfentials (unrestricted) > */
                /*  ID for this is gcp-key */
                /*  variable: name it anything you want in this case is 'GOOGLE_APPLICATION_CREDENTIALS' */
                /*  Will use 'gcp-key' stored in Jenkins Credential and will name it as 'GOOGLE_APPLICATION_CREDENTIALS' */
                /* Ensure that Gcloud is available in the path export PATH*/
                /* Activate service account */
                /* Set the project on where it would execute the commands*/
                /* Configure Docker with GCR */
                /* Build the Docker image -> google cloud registry URL with the project name/ project_image_name (in this case the project image name is ml-project:latest */
                /* Push the Docker Image to GCR */
                withCredentials([file(credentialsId: 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Building and Pushing Docker Image to GCR.............'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                        gcloud config set project ${GCP_PROJECT}

                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

                        docker push gcr.io/${GCP_PROJECT}/ml-project:latest 

                        '''
                    }
                }
            }
        }
        
    }
}