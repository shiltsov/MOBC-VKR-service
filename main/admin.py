import os
from django.contrib import admin, messages
from .models import LogEntry, UploadedDatasetFile, TaskStatus
from django.utils.html import format_html
from django.urls import reverse, path
from django.http import HttpResponseRedirect

from .tasks import train_model


@admin.register(LogEntry)
class LogEntryAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user_input")
    readonly_fields = ("created_at", "user_input", "result_json")
    search_fields = ("user_input",)

@admin.action(description="Удалить выбранные файлы и записи")
def delete_files_and_records(modeladmin, request, queryset):
    for obj in queryset:
        obj.delete()  # вызывает твой метод delete(), удаляет и запись, и файл

# статус задания в очереди сообщений
@admin.register(TaskStatus)
class TaskStatusAdmin(admin.ModelAdmin):
    list_display = ("task_id", "status_colored", "created_at", "updated_at")
    readonly_fields = ("task_id", "status", "created_at", "updated_at")
    ordering = ("-created_at",)
    search_fields = ("task_id", "status")

    def status_colored(self, obj):
        color_map = {
            "pending": "gray",
            "running": "orange",
            "completed": "green",
            "failed": "red",
        }
        color = color_map.get(obj.status, "black")
        return format_html('<span style="color: {};">{}</span>', color, obj.status)

    status_colored.short_description = "Status"

# подгружаемые датасеты
@admin.register(UploadedDatasetFile)
class UploadedDatasetFileAdmin(admin.ModelAdmin):
    list_display = ("file_link", "edit_link")
    # чтобы попороть удаление файлов для bulk_delete
    actions = [delete_files_and_records]

    def get_actions(self, request):
        actions = super().get_actions(request)
        if 'delete_selected' in actions:
            del actions['delete_selected']
        return actions

    def file_link(self, obj):
        return format_html('<a href="{}" download>{}</a>', obj.file.url, obj.file.name)
    file_link.short_description = "Файл"

    def edit_link(self, obj):

        link = reverse("admin:main_uploadeddatasetfile_change", args=[obj.pk])
        return format_html('<a href="{}">Редактировать</a>', link)
    edit_link.short_description = "Редактировать"


    ##################
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('train-model/', self.admin_site.admin_view(self.train_model_view), name='train-model'),
        ]
        return custom_urls + urls


    def train_model_view(self, request):
        if request.method == "POST":
            #try:
            epochs = int(request.POST.get("epochs", 1))


            task = TaskStatus.objects.create(status="Pending")

            # отправляем задачу в очередь
            train_model.delay(task.task_id, epochs)

            #    self.message_user(request, f"Модель отправлена на обучение, эпох: {epochs}", messages.SUCCESS)
            #except Exception as e:
            #    self.message_user(request, f"Ошибка обучения: {e}", messages.ERROR)
        return HttpResponseRedirect(reverse("admin:main_uploadeddatasetfile_changelist"))

    def changelist_view(self, request, extra_context=None):
        if extra_context is None:
            extra_context = {}
        extra_context["train_model_form"] = True
        return super().changelist_view(request, extra_context=extra_context)


