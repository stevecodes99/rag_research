from paperswithcode import PapersWithCodeClient

def list_all_areas():
    client = PapersWithCodeClient()
    areas = client.area_list()
    return areas

def list_tasks_for_area(client, area_id):
    tasks = client.area_task_list(area_id)
    return tasks

# Example usage
client = PapersWithCodeClient()
areas = list_all_areas()

for area in areas.results:
    print(f"Area: {area.name}, Area ID: {area.id}")
    tasks = list_tasks_for_area(client, area.id)
    for task in tasks.results:
        print(f"  Task: {task.name}, Task ID: {task.id}")


def list_all_paper():
    client = PapersWithCodeClient()
    tasks = client.task_paper_list('question-answering')
    return tasks

papers = list_all_paper()
print(len(tasks.results))
for paper in papers.results:
    print(f"paper Name: {paper.title}, paper ID: {task.authors}")