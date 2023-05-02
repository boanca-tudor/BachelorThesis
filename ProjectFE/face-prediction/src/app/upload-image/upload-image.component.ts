import { Component, OnInit } from '@angular/core';
import { HttpEventType, HttpResponse } from '@angular/common/http';
import { Observable } from 'rxjs';
import { UploadService } from './upload.service';

@Component({
  selector: 'app-upload-image',
  templateUrl: './upload-image.component.html',
  styleUrls: ['./upload-image.component.css'],
})
export class UploadImageComponent implements OnInit {
  currentFile?: File;
  progress = 0;
  message = '';
  preview = '';

  information?: Observable<any>;

  constructor(private uploadService: UploadService) {}

  selectFile(event: any): void {
    this.message = '';
    this.preview = '';
    this.progress = 0;
    this.currentFile = event.target.files[0];
    console.log(this.currentFile);
  }

  upload(): void {
    this.progress = 0;

    if (this.currentFile) {
      this.uploadService.upload(this.currentFile).subscribe({
        next: (event: any) => {
          if (event.type === HttpEventType.UploadProgress) {
            this.progress = Math.round((100 * event.loaded) / event.total);
          } else if (event instanceof HttpResponse) {
            this.message = event.body.message;
            this.information = this.uploadService.getFiles();
            this.progress = 0;
          }
        },
        error: (err: any) => {
          console.log(err);
          this.progress = 0;

          if (err.error && err.error.message) {
            this.message = err.error.message;
          } else {
            this.message = 'Uploading failed!';
          }

          this.currentFile = undefined;
        },
      });
    }
  }

  ngOnInit(): void {
    this.information = this.uploadService.getFiles();
  }
}
