import labelbox as lb
import urllib


def download_video_data(api_key, project_id):
    lb_client = lb.Client(api_key=api_key)
    lb_project = lb_client.get_project(project_id=project_id)
    
    labels = lb_project.video_label_generator()


    
if __name__ == "__main__":

    LABELBOX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3k0MG9pNjkyZjYwMHplcGdlM3o2anpyIiwib3JnYW5pemF0aW9uSWQiOiJja3k0MG9ocXEyZjV6MHplcGVsYjk1N3YyIiwiYXBpS2V5SWQiOiJja3lieXAxdnUwOThnMHpieDNycmdjMThiIiwic2VjcmV0IjoiY2M0YTUzYTA2MmYxMDM4NmY1MWJiNTExM2Q1YTgxYTQiLCJpYXQiOjE2NDIwMTcyODUsImV4cCI6MjI3MzE2OTI4NX0.kXsSSgzrAeFdYdryYgzdok6eiyHydLA88ZP_Pd7EnuQ"
    LABELBOX_PROJECT_ID = "cky4nw7aaohqu0zdh6d75gobs"
    
    download_video_data(api_key=LABELBOX_API_KEY, project_id=LABELBOX_PROJECT_ID)
    
