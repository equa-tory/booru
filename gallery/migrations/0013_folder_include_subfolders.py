from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gallery', '0012_folder_parent'),
    ]

    operations = [
        migrations.AddField(
            model_name='folder',
            name='include_subfolders',
            field=models.BooleanField(default=False),
        ),
    ]
