import { Component, OnInit } from '@angular/core';
import { HttpEventType, HttpResponse } from '@angular/common/http';
import { Observable } from 'rxjs';
import { UploadService } from './upload.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-upload-image',
  templateUrl: './upload-image.component.html',
  styleUrls: ['./upload-image.component.css'],
})
export class UploadImageComponent implements OnInit {
  currentFile?: File;
  progress = 0;
  showError: boolean;
  errorMessage: string;

  information?: Observable<any>;

  constructor(private uploadService: UploadService, private router: Router) {
    this.showError = false;
    this.errorMessage = '';
  }

  selectFile(event: any): void {
    this.progress = 0;
    this.showError = false;
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
            this.information = this.uploadService.getFiles();
            this.progress = 0;
            this.router.navigate(['generate']);
          }
        },
        error: (err: any) => {
          console.log(err);
          this.progress = 0;

          this.showError = true;
          if (err.error && err.error.message) {
            this.errorMessage = err.error.message;
          } else {
            this.errorMessage = 'Uploading failed!';
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
